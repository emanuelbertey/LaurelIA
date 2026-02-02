#![recursion_limit = "256"]

/*!
Cortex Thread: Ensemble of 3 sLSTM models with Multi-threaded Training and Generation.
Uses std::thread::scope for parallel processing and shared memory.
*/

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use anyhow::Result;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::collections::HashSet;
use std::time::Instant;
use std::sync::{Arc, Barrier};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::AddedToken;

use xlstm::{LstmType, XLstm, XLstmconfig, LSTMState};
use rand::Rng;

/// Profesional Tokenizer using Hugging Face 'tokenizers'
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self> {
        let special_tokens_strings = vec![
            "[ENG]".to_string(),
            "[SEP]".to_string(),
            "[ESP]".to_string(),
            "[EOS]".to_string(),
            "<PAD>".to_string(),
        ];

        let special_tokens: Vec<AddedToken> = special_tokens_strings
            .iter()
            .map(|t| AddedToken::from(t, true))
            .collect();

        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut tokenizer = HFTokenizer::new(model);

        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            ' ',
            PrependScheme::Always,
            true,
        )));

        let mut alphabet = HashSet::new();
        alphabet.insert('\n');
        alphabet.insert(' ');

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(alphabet)
            .special_tokens(special_tokens.clone())
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);

        let temp_file = "temp_train_cortex_thread.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| anyhow::anyhow!(e))?;
        fs::remove_file(temp_file)?;

        for token in special_tokens_strings {
            tokenizer.add_special_tokens(&[AddedToken::from(token, true)]);
        }

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.tokenizer.save(path, true)
            .map_err(|e| anyhow::anyhow!("Error saving: {}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Error loading: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

fn create_batch(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + (i * stride); 
        for j in 0..seq_length {
            if current_start + j + 1 < tokens.len() {
                x_indices.push(tokens[current_start + j] as u32);
                y_indices.push(tokens[current_start + j + 1] as u32);
            } else {
                x_indices.push(0); 
                y_indices.push(0);
            }
        }
    }

    let x = Tensor::from_vec(x_indices, (batch_size, seq_length), device)?;
    let y = Tensor::from_vec(y_indices, (batch_size, seq_length), device)?;

    Ok((x, y))
}

/// Devuelve los top-5 LOGITS procesados
pub fn get_top_5_logits(logits: &Tensor) -> Result<Vec<(usize, f32)>> {
    let logits = logits.squeeze(0)?; // [vocab_size]
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    
    let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(indexed.into_iter().take(5).collect())
}

/// Muestreo Top-K con voting. Los logits ya vienen promediados o listos.
fn sample_from_logits(averaged_logits: &Tensor, temperature: f32) -> Result<usize> {
    let scaled_logits = (averaged_logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits.squeeze(0)?, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    
    let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = 5; 
    let top_k_probs = &indexed[..k.min(indexed.len())];
    
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng(); 
    let mut sample: f32 = rng.random::<f32>() * sum;

    for (i, &p) in weights.iter().enumerate() {
        if sample <= p {
            return Ok(indices[i]);
        }
        sample -= p;
    }

    Ok(indices[0])
}

/// Generación EN PARALELO de los 3 modelos
fn generate_threaded_text(
    models: &[XLstm],
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &Device,
    barrier: &Arc<Barrier>, // Semáforo/Barrera para sincronizar hilos
) -> Result<String> {
    let mut current_text = seed_text.to_string();
    let seed_tokens = tokenizer.encode(seed_text);
    
    if seed_tokens.is_empty() {
        return Ok(current_text);
    }

    let mut current_states: Vec<Option<Vec<Option<LSTMState>>>> = vec![None; models.len()];
    let mut current_tokens = seed_tokens;

    for i in 0..length {
        let tokens_to_process = if i == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };

        let seq_len = tokens_to_process.len();
        let indices_vec: Vec<u32> = tokens_to_process.iter().map(|&t| t as u32).collect();
        let input = Tensor::from_vec(indices_vec, (1, seq_len), device)?;

        let mut model_results = Vec::with_capacity(models.len());
        
        // --- PROCESAMIENTO MULTI-HILO ---
        let scope_res: Result<()> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            
            for (m_idx, model) in models.iter().enumerate() {
                let input_ref = &input;
                let state = current_states[m_idx].take();
                let b = barrier.clone();
                
                handles.push(s.spawn(move || {
                    // Sincronizar antes de empezar (comprobar que hilos están libres)
                    b.wait();
                    
                    let (output, next_state) = model.forward(input_ref, state)?;
                    let last_step_logits = output.narrow(1, seq_len - 1, 1)?.squeeze(1)?.detach();
                    
                    // Sincronizar al terminar
                    b.wait();
                    
                    Ok::<(Tensor, Vec<Option<LSTMState>>), anyhow::Error>((last_step_logits, next_state))
                }));
            }
            
            for handle in handles {
                let result = handle.join().unwrap()?;
                model_results.push(result);
            }
            Ok(())
        });
        scope_res?;

        // Procesar los resultados obtenidos en orden
        let mut model_logits = Vec::with_capacity(model_results.len());
        for (m_idx, (logits, next_state)) in model_results.into_iter().enumerate() {
            model_logits.push(logits);
            current_states[m_idx] = Some(next_state.into_iter().map(|s| s.map(|state| state.detach())).collect());
        }

        // Promediar logits (Voting)
        let mut combined_logits = model_logits[0].clone();
        for j in 1..model_logits.len() {
            combined_logits = (combined_logits + &model_logits[j])?;
        }
        let averaged_logits = (combined_logits / (model_logits.len() as f64))?;

        let next_token = sample_from_logits(&averaged_logits, 0.8)?;
        current_tokens.push(next_token);
        
        if let Some(t) = tokenizer.id_to_token(next_token) {
            let mut clean_token = t.clone();
            if clean_token.contains('Ċ') || clean_token.contains('Ġ') {
               clean_token = clean_token.replace("Ċ", "\n").replace("Ġ", " ");
            }
            current_text.push_str(&clean_token);
        }
    }

    Ok(current_text)
}

fn main() -> Result<()> {
    println!("Cortex Thread: Ensemble de 3 Modelos con HILOS y Semáforos");
    println!("==========================================================\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin cortexthread -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "cortex_tokenizer.json";
    let target_vocab_size = 1024;
    let device = Device::Cpu;

    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    let vocab_size = tokenizer.vocab_size();
    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);

    let hidden_size = 256; 
    let num_layers = 1;
    let num_blocks = 1;
    let output_size = vocab_size; 
    let seq_length = 128; 
    let batch_size = 16; 
    let stride = 128;     
    let num_epochs = 20;

    let config = XLstmconfig::new(hidden_size, hidden_size, num_layers, num_blocks, output_size)
        .with_vocab_size(vocab_size)
        .with_lstm_type(LstmType::SLSTM) 
        .with_use_projection(true);

    let mut models = Vec::new();
    let mut varmaps = Vec::new();
    let mut optimizers = Vec::new();

    for i in 1..=3 {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = config.init(vb)?;
        
        let mut params = Vec::new();
        {
            let data = vm.data().lock().unwrap();
            for (_, var) in data.iter() {
                params.push(var.clone());
            }
        }
        let optimizer = AdamW::new(params, ParamsAdamW { lr: 2e-3, ..Default::default() })?;
        
        models.push(model);
        varmaps.push(vm);
        optimizers.push(optimizer);
        
        let m_path = format!("cortex_model_{}.safetensors", i);
        if Path::new(&m_path).exists() {
            varmaps[i-1].load(&m_path)?;
        }
    }

    // Barrera para sincronizar los 3 hilos (+ el principal si fuera necesario, pero aquí usamos 3 para los modelos)
    let thread_barrier = Arc::new(Barrier::new(3));

    let num_actual_sequences = tokens.len().saturating_sub(seq_length) / stride;
    let num_batches = num_actual_sequences / batch_size;

    println!("Iniciando entrenamiento MULTI-HILO (3 WorkThreads)...");

    for epoch in 0..num_epochs {
        let mut total_losses = vec![0.0f32; 3];
        let start_time = Instant::now();

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size * stride;
            let (input_batch, target_batch) = create_batch(&tokens, start_idx, batch_size, seq_length, stride, &device)?;
            let target_flat = target_batch.reshape((batch_size * seq_length,))?;
            
            // Compartir Batch entre hilos
            let input_ref = &input_batch;
            let target_ref = &target_flat;

            // --- ENTRENAMIENTO EN PARALELO ---
            std::thread::scope(|s| {
                let mut handles = Vec::new();
                
                // Iterar sobre modelos (lectura) y optimizadores (escritura) de forma segura
                let zip_iter = models.iter().zip(optimizers.iter_mut());

                for (m_idx, (model, optimizer)) in zip_iter.enumerate() {
                    let b = &thread_barrier;
                    
                    handles.push(s.spawn(move || {
                        b.wait(); // Esperar a que todos estén listos

                        let (logits, _) = model.forward(input_ref, None)?;
                        let logits_flat = logits.reshape((batch_size * seq_length, vocab_size))?;
                        let loss = candle_nn::loss::cross_entropy(&logits_flat, target_ref)?;
                        let grads = loss.backward()?;
                        optimizer.step(&grads)?;
                        
                        b.wait(); // Esperar a que todos terminen el heavy lift
                        Ok::<f32, anyhow::Error>(loss.to_scalar::<f32>()?)
                    }));
                }
                
                for (m_idx, handle) in handles.into_iter().enumerate() {
                    total_losses[m_idx] += handle.join().unwrap().unwrap();
                }
            });

            if batch_idx % 10 == 0 {
                print!("\rEpoch {}/{} | Batch {}/{} | L1={:.3} L2={:.3} L3={:.3}", 
                    epoch+1, num_epochs, batch_idx, num_batches,
                    total_losses[0] / (batch_idx+1) as f32,
                    total_losses[1] / (batch_idx+1) as f32,
                    total_losses[2] / (batch_idx+1) as f32);
                io::stdout().flush()?;
            }
        }

        println!("\nEpoch {} OK ({:.1}s). Guardando...", epoch+1, start_time.elapsed().as_secs_f32());
        for i in 0..3 { varmaps[i].save(&format!("cortex_model_{}.safetensors", i+1))?; }

        // Generar con ensemble hilos
        let mut rng = rand::rng();
        let ridx = rng.random_range(0..tokens.len() - 10);
        let seed = tokenizer.decode(&tokens[ridx..ridx+5]);
        println!("  Seed: '{}'", seed);
        let gen_sample = generate_threaded_text(&models, &tokenizer, &seed, 50, &device, &thread_barrier)?;
        println!("  Ensemble Thread: {}\n", gen_sample);
    }

    // Modo interactivo
    println!("\n--- Modo Interactivo (Cortex Thread Ensemble) ---");
    loop {
        print!("Input > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input == "exit" { break; }
        
        let gen_interactive = generate_threaded_text(&models, &tokenizer, input, 100, &device, &thread_barrier)?;
        println!("Cortex (Threaded): {}\n", gen_interactive);
    }

    Ok(())
}
