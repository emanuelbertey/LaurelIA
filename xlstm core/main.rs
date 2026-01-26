mod xlstm_impl {
    pub mod candle {
        pub use candle_core::*;
        pub use candle_nn::Module;
    }
    // Incluimos el código en su propio módulo para que los //! sean válidos
    // y no colisionen los imports de Result/VarBuilder.
    include!("xlstm.rs");
}

// Re-exportamos lo necesario para el entrenamiento
use xlstm_impl::{Config, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};

use anyhow::Context;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::collections::HashSet;
use std::time::Instant;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use rand::Rng;

/// Tokenizador (exactamente de slstmchat.rs)
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> anyhow::Result<Self> {
        let model = BPE::builder().byte_fallback(true).build().map_err(|e| anyhow::anyhow!(e))?;
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(' ', PrependScheme::Always, true)));
        let mut alphabet = HashSet::new();
        alphabet.insert('\n'); alphabet.insert(' ');
        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(0)
            .initial_alphabet(alphabet)
            .build();
        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train_core.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()]).map_err(|e| anyhow::anyhow!(e))?;
        fs::remove_file(temp_file)?;
        Ok(Self { tokenizer })
    }
    pub fn save(&self, path: &str) -> anyhow::Result<()> { self.tokenizer.save(path, true).map_err(|e| anyhow::anyhow!(e)) }
    pub fn load(path: &str) -> anyhow::Result<Self> { Ok(Self { tokenizer: HFTokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))? }) }
    pub fn encode(&self, text: &str) -> Vec<usize> { self.tokenizer.encode(text, false).unwrap().get_ids().iter().map(|&id| id as usize).collect() }
    pub fn decode(&self, indices: &[usize]) -> String { let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect(); self.tokenizer.decode(&u32_indices, true).unwrap() }
    pub fn vocab_size(&self) -> usize { self.tokenizer.get_vocab_size(true) }
    pub fn id_to_token(&self, id: usize) -> Option<String> { self.tokenizer.id_to_token(id as u32) }
}

fn create_batch(tokens: &[usize], start_idx: usize, batch_size: usize, seq_length: usize, stride: usize, device: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);
    for i in 0..batch_size {
        let current_start = start_idx + (i * stride); 
        for j in 0..seq_length {
            if current_start + j + 1 < tokens.len() {
                x_indices.push(tokens[current_start + j] as u32);
                y_indices.push(tokens[current_start + j + 1] as u32);
            } else { x_indices.push(0); y_indices.push(0); }
        }
    }
    let x = Tensor::from_vec(x_indices, (batch_size, seq_length), device)?;
    let y = Tensor::from_vec(y_indices, (batch_size, seq_length), device)?;
    Ok((x, y))
}

fn sample_from_logits(logits: &Tensor, temperature: f32) -> anyhow::Result<usize> {
    let scaled_logits = (logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut probs_indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = 10;
    let top_k_probs = &probs_indexed[..k.min(probs_indexed.len())];
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    let sum: f32 = weights.iter().sum();
    let sample = rand::rng().random::<f32>() * sum;
    let mut acc = 0.0;
    for (i, &p) in weights.iter().enumerate() { acc += p; if sample <= acc { return Ok(indices[i]); } }
    Ok(indices[0])
}

fn generate_text(model: &Model, tokenizer: &Tokenizer, seed_text: &str, length: usize, device: &Device) -> anyhow::Result<String> {
    let mut current_text = seed_text.to_string();
    let seed_tokens = tokenizer.encode(seed_text);
    if seed_tokens.is_empty() { return Ok(current_text); }
    let mut state = model.new_state(1, device)?;
    let mut last_logits = None;
    for &token in &seed_tokens {
        let input = Tensor::new(&[token as u32], device)?;
        last_logits = Some(model.forward(&input, &mut state)?);
    }
    for _ in 0..length {
        let logits = last_logits.as_ref().context("No logits")?.squeeze(0)?;
        let next_token = sample_from_logits(&logits, 0.8)?;
        if let Some(t) = tokenizer.id_to_token(next_token) {
            let mut clean_token = t.clone();
            if clean_token.contains('Ċ') || clean_token.contains('Ġ') { clean_token = clean_token.replace("Ċ", "\n").replace("Ġ", " "); }
            current_text.push_str(&clean_token);
        }
        let input = Tensor::new(&[next_token as u32], device)?;
        last_logits = Some(model.forward(&input, &mut state)?);
    }
    Ok(current_text)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Uso: cargo run --bin xlstm_core_train -- input.txt"); std::process::exit(1); }
    let text_file = &args[1];
    let device = Device::Cpu;

    // Params slstmchat.rs
    let target_vocab_size = 1024;
    let hidden_size = 256; 
    let num_blocks = 3;
    let seq_length = 128; 
    let batch_size = 16; 
    let stride = 128;     
    let num_heads = 16;
    let num_epochs = 50;

    let tokenizer_path = "core_tokenizer.json";
    let tokenizer = if Path::new(tokenizer_path).exists() { Tokenizer::load(tokenizer_path)? }
    else {
        let text = fs::read_to_string(text_file)?;
        let t = Tokenizer::from_text(&text, target_vocab_size)?;
        t.save(tokenizer_path)?; t
    };

    let vocab_size = tokenizer.vocab_size();
    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);

    let config = Config {
        vocab_size, embedding_dim: hidden_size, num_blocks, num_heads,
        head_dim: hidden_size / num_heads, qk_dim_factor: 0.5,
         v_dim_factor: 1.0,
        ffn_proj_factor: 2.667,
         ffn_round_up_to_multiple_of: 64, 
         mlstm_round_up_to_multiple_of: 64,
        norm_eps: 1e-6, 
        cell_norm_eps: 1e-6, 
        gate_soft_cap: 25.0, //gate_soft_cap: 15.0,
        output_logit_soft_cap: 40.0,// output_logit_soft_cap: 30.0,
        chunk_size: 64, 
        add_post_blocks_norm: true, 
        tie_word_embeddings: false, 
        use_bias: false,
        bos_token_id: 0, 
        eos_token_id: 0, 
        pad_token_id: 0,
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&config, vb)?;

    let model_path = "core_model.safetensors";
    let existe_modelo = Path::new(model_path).exists();
    let mut continuar_entrenamiento = !existe_modelo;

    if existe_modelo {
        varmap.load(model_path)?;
        print!("¿Deseas seguir entrenando el modelo cargado? (s/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "s" { continuar_entrenamiento = true; }
    }

    if continuar_entrenamiento {
        let mut optimizer = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 1e-4, ..Default::default() })?;
        let num_actual_sequences = tokens.len().saturating_sub(seq_length).div_ceil(stride);
        let num_batches = num_actual_sequences / batch_size;

        println!("Iniciando entrenamiento...");
        for epoch in 0..num_epochs {
            let mut total_loss = 0.0f32;
            let mut correct = 0;
            let mut total = 0;
            for batch_idx in 0..num_batches {
                let start_time = Instant::now();
                let current_batch_start_seq = batch_idx * batch_size;
                let (input_batch, target_batch) = create_batch(&tokens, current_batch_start_seq * stride, batch_size, seq_length, stride, &device)?;

                let mut state = model.new_state(batch_size, &device)?;
                let mut logits_list = Vec::with_capacity(seq_length);
                for i in 0..seq_length {
                    let step_input = input_batch.narrow(1, i, 1)?.squeeze(1)?;
                    let step_logits = model.forward(&step_input, &mut state)?;
                    logits_list.push(step_logits.unsqueeze(1)?);
                }
                let logits_seq = Tensor::cat(&logits_list, 1)?;
                let logits_flat = logits_seq.reshape((batch_size * seq_length, vocab_size))?;
                let target_flat = target_batch.reshape((batch_size * seq_length,))?;
                let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
                optimizer.step(&loss.backward()?)?;

                total_loss += loss.to_scalar::<f32>()?;
                let preds = logits_flat.argmax(1)?;
                correct += preds.eq(&target_flat)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
                total += batch_size * seq_length;

                if batch_idx % 1 == 0 {
                    print!("\r  Epoch {} [{}/{}] Loss: {:.4} Acc: {:.2}% ({:.1}s)", 
                        epoch + 1, batch_idx + 1, num_batches, total_loss / (batch_idx+1) as f32,
                        100.0 * correct as f32 / total as f32, start_time.elapsed().as_secs_f32());
                    io::stdout().flush().unwrap();
                }
            }
            println!();
            varmap.save(model_path)?;
            if let Ok(sample) = generate_text(&model, &tokenizer, "The ", 50, &device) {
                println!("  Muestra: {}\n", sample);
            }
        }
    }

    // Modo interactivo (Inferencia)
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO - GENERACIÓN DE TEXTO         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    let mut gen_length: usize = 200;
    loop {
        print!("Semilla > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "salir" { break; }
        if input.to_lowercase().starts_with("len") {
            if let Some(n) = input.split_whitespace().nth(1).and_then(|s| s.parse::<usize>().ok()) {
                gen_length = n; println!("Nueva longitud: {} tokens\n", gen_length); continue;
            }
        }
        println!("Generando...");
        match generate_text(&model, &tokenizer, input, gen_length, &device) {
            Ok(res) => println!("Resultado: {}\n", res),
            Err(e) => println!("Error: {:?}\n", e),
        }
    }
    Ok(())
}
