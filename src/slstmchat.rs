#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization

This example demonstrates how to use xLSTM for text generation
using a simple character-level tokenizer that can be saved/loaded as JSON.

*/

use candle_core::{Device, Tensor, Result, Module, DType, IndexOp};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW, ops::softmax};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::{LstmType, XLstm, XLstmconfig, BlockType};
use rand::Rng;

/// Tokenizador profesional usando la librería 'tokenizers' de Hugging Face
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    /// Crea un nuevo tokenizador BPE entrenado desde un texto
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::new(BPE::default());
        
        // Usar pre-tokenizador de espacios en blanco
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .build();

        // Envolver el entrenador de manera genérica usando el trait From
        let mut trainer_wrapper = TrainerWrapper::from(trainer);

        // Entrenar desde el archivo temporal
        let temp_file = "temp_train.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    /// Guarda el tokenizador en un archivo
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true)
            .map_err(|e| format!("Error al guardar: {}", e))?;
        println!("Tokenizador guardado en: {}", path);
        Ok(())
    }

    /// Carga el tokenizador desde un archivo
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| format!("Error al cargar: {}", e))?;
        println!("Tokenizador cargado desde: {}", path);
        Ok(Self { tokenizer })
    }

    /// Convierte texto a índices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    /// Convierte índices a texto
    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }

    /// Obtiene el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Obtiene el string de un token por su índice
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

/// Crea un batch de entrenamiento (one-hot) de forma eficiente usando una matriz identidad
fn create_batch(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    vocab_size: usize,
    device: &Device,
    eye: &Tensor,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i;
        for j in 0..seq_length {
            x_indices.push(tokens[current_start + j] as u32);
            y_indices.push(tokens[current_start + j + 1] as u32);
        }
    }

    let x_indices_tensor = Tensor::from_vec(x_indices, (batch_size * seq_length, ), device)?;
    
    // Select input from eye matrix: [batch_size * seq_length, vocab_size]
    let x = eye.index_select(&x_indices_tensor, 0)?
               .reshape((batch_size, seq_length, vocab_size))?;

    let y = Tensor::from_vec(y_indices, (batch_size, seq_length), device)?; // Targets are indices

    Ok((x, y))
}


/// Selecciona un token usando muestreo estocástico con Top-K y temperatura
fn sample_from_logits(logits: &Tensor, temperature: f32) -> Result<usize, Box<dyn Error>> {
    // logits: [1, vocab_size]
    let logits = logits.squeeze(0)?;
    let vocab_size = logits.dim(0)?;
    
    // To Vec for manual sampling (simpler than tensor operations for top-k sampling for now)
    let logits_vec = logits.to_vec1::<f32>()?;
    
    let mut probs_vec: Vec<(usize, f32)> = logits_vec.iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    // Softmax is applied later in sampling logic or here?
    // The original code applied softmax first, then top-k on probs.
    // Let's replicate logic: Softmax then Top-K.
    
    // We can do softmax on tensor first
    let probs = softmax(&logits, 0)?;
    let probs_vec_tensor = probs.to_vec1::<f32>()?;
    let mut probs_indexed: Vec<(usize, f32)> = probs_vec_tensor.into_iter().enumerate().collect();


    // --- TOP-K ---
    // Ordenar de mayor a menor probabilidad
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Solo nos quedamos con los 5 o 10 mejores candidatos
    let k = 5; 
    let top_k_probs = &probs_indexed[..k.min(vocab_size)];
    
    // Extraer solo los pesos para el muestreo
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    // --------------------

    // Si la temperatura es muy baja, actuar de forma determinista (Greedy)
    if temperature <= 1e-6 {
        return Ok(indices[0]);
    }

    // Aplicar temperatura sobre el Top-K
    for p in weights.iter_mut() {
        // p is probability. log(p)/temp -> exp
         *p = (p.max(1e-10).ln() / temperature).exp();
    }

    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng(); 
    let sample: f32 = rng.random::<f32>() * sum;
    let mut cumulative = 0.0;

    for (i, &p) in weights.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative {
            return Ok(indices[i]);
        }
    }

    Ok(indices[0])
}

/// Genera texto de forma recurrente manteniendo el estado interno del modelo
fn generate_text(
    model: &XLstm,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<String, Box<dyn Error>> {
    let mut current_text = seed_text.to_string();
    let seed_tokens = tokenizer.encode(seed_text);
    
    if seed_tokens.is_empty() {
        return Ok(current_text);
    }

    let eye = Tensor::eye(vocab_size, DType::F32, device)?;
    
    let mut current_state = None; 
    let mut current_tokens = seed_tokens;

    for i in 0..length {
        let tokens_to_process = if i == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };

        let seq_len = tokens_to_process.len();
        
        let indices_tensor = Tensor::from_vec(
            tokens_to_process.iter().map(|&t| t as u32).collect::<Vec<_>>(), 
            (seq_len,), 
            device
        )?;

        let input = eye.index_select(&indices_tensor, 0)?
            .reshape((1, seq_len, vocab_size))?;

        let (output, next_state) = model.forward(&input, current_state)?;
        current_state = Some(next_state);

        let (_b, _l, _v) = output.dims3()?;
        // Extract last step logits
        let last_logits = output.narrow(1, seq_len - 1, 1)?
            .squeeze(1)?; // [1, vocab_size]

        let next_token = sample_from_logits(&last_logits, 0.8)?;

        current_tokens.push(next_token);
        if let Some(t) = tokenizer.id_to_token(next_token) {
            let clean_token = t
                .replace("Ċ", "\n") 
                .replace("Ġ", " "); 
            
            current_text.push_str(&clean_token);
        }
    }

    Ok(current_text)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM (SLSTM) Text Generation con Tokenizador (Candle)");
    println!("====================================================\n");

    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin slstmchat -- <archivo.txt>");
        eprintln!("Ejemplo: cargo run --bin slstmchat -- input.txt");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer_slstm.json";
    let model_path = "xlstm_chat_model_slstm.safetensors";

    let target_vocab_size = 2048;

    let tokenizer = if Path::new(tokenizer_path).exists() {
        println!("Cargando tokenizador existente...");
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Entrenando nuevo tokenizador profesional (BPE) desde {}...", text_file);
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    println!("Tamaño del vocabulario: {}\n", tokenizer.vocab_size());

    println!("Cargando texto de entrenamiento...");
    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    println!("Tokens totales: {}\n", tokens.len());

    let vocab_size = tokenizer.vocab_size();
    let hidden_size = 256; 
    let num_layers = 1;
    let num_blocks = 3;
    let output_size = vocab_size; 
    let dropout = 0.1;

    let seq_length = 128; 
    let batch_size = 16; 
    let stride = 64;     
    let num_epochs = 50;
    let num_heads = 2; // Not used for sLSTM but config requires it

    println!("Configuración del modelo:");
    println!("  Bloques: {}", num_blocks);
    println!("  Hidden size: {}", hidden_size);
    println!("  Seq length: {}", seq_length);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}\n", num_epochs);

    let device = Device::Cpu;
    let eye = Tensor::eye(vocab_size, DType::F32, &device)?;

     let config = XLstmconfig::new(vocab_size, hidden_size, num_layers, num_blocks, output_size)
        .with_dropout(dropout)
        .with_num_heads(num_heads)
        .with_lstm_type(LstmType::SLSTM) // FORCED SLSTM
        .with_use_projection(true);   

    let model_file_path = Path::new(model_path);
    let existe_modelo = model_file_path.exists();
    
    let mut continuar_entrenamiento = false;
    if existe_modelo {
        print!("¿Deseas seguir entrenando el modelo cargado? (s/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "s" {
            continuar_entrenamiento = true;
        }
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = config.init(vb)?;

    if existe_modelo {
         if !continuar_entrenamiento {
             println!("¡Modelo encontrado! Cargando pesos para generación...");
             varmap.load(model_path)?;
             println!("Modelo cargado exitosamente!\n");
         } else {
             println!("Cargando modelo previo para continuar entrenamiento...");
             varmap.load(model_path)?;
         }
    } else {
         println!("No se encontró modelo guardado. Iniciando entrenamiento desde cero...\n");
    }
    
    // Logic from main.rs training loop, adapted
    if !existe_modelo || continuar_entrenamiento {

        if !tokens.is_empty() {
             let first_token_idx = tokens[0];
             let first_token_str = tokenizer.id_to_token(first_token_idx).unwrap_or("?".to_string());
             println!("--- INSPECCIÓN DE EMBEDDING ---");
             println!("  Token Index: {}", first_token_idx);
             println!("  Token Str: '{}'", first_token_str);
             println!("-----------------------------\n");
        }

        let num_sequences = tokens.len().saturating_sub(seq_length);
        let num_actual_sequences = (num_sequences + stride - 1) / stride;

        // Group parameters for optimizers (as in main.rs)
        let parsed_block_types = match config.lstm_type {
            LstmType::SLSTM => vec![BlockType::SLSTM; num_blocks],
            LstmType::MLSTM => vec![BlockType::MLSTM; num_blocks],
            LstmType::Alternate => (0..num_blocks)
                .map(|i| if i % 2 == 0 { BlockType::SLSTM } else { BlockType::MLSTM })
                .collect(),
            LstmType::Custom(ref types) => types.clone(),
        };

        let mut slstm_params = Vec::new();
        let mut mlstm_params = Vec::new();
        let mut other_params = Vec::new();

        for (name, var) in varmap.data() {
            if name.starts_with("block_") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Some(block_part) = parts.first() {
                    if let Some(idx_str) = block_part.strip_prefix("block_") {
                         if let Ok(idx) = idx_str.parse::<usize>() {
                             if idx < parsed_block_types.len() {
                                 match parsed_block_types[idx] {
                                     BlockType::SLSTM => slstm_params.push(var),
                                     BlockType::MLSTM => mlstm_params.push(var),
                                 }
                             } else { other_params.push(var); }
                         } else { other_params.push(var); }
                    } else { other_params.push(var); }
                } else { other_params.push(var); }
            } else { other_params.push(var); }
        }

        let mut optim_slstm = AdamW::new(slstm_params, ParamsAdamW { lr: 1e-4, ..Default::default() })?;
        let mut optim_mlstm = AdamW::new(mlstm_params, ParamsAdamW { lr: 1e-5, ..Default::default() })?;
        let mut optim_other = AdamW::new(other_params, ParamsAdamW { lr: 1e-4, ..Default::default() })?;

        println!("Iniciando entrenamiento...\n");

        let num_batches = num_actual_sequences.div_ceil(batch_size);

        for epoch in 0..num_epochs {
            let mut total_loss = 0.0f32;
            let mut num_losses = 0;
            let mut correct = 0;
            let mut total = 0;
            let mut current_state = None;

            for batch_idx in 0..num_batches {
                let current_batch_start_seq = batch_idx * batch_size;
                let current_batch_size = (batch_size).min(num_actual_sequences - current_batch_start_seq);

                if current_batch_size == 0 { break; }
                if current_batch_size < batch_size { break; } // Skip incomplete

                let (input_batch, target_batch) = create_batch(
                    &tokens,
                    current_batch_start_seq * stride,
                    current_batch_size,
                    seq_length,
                    vocab_size,
                    &device,
                    &eye,
                )?;

                // Warm up state if batch 0
                 if batch_idx == 0 {
                    let (_, warm_state) = model.predict_last(&input_batch, None)?;
                    current_state = Some(warm_state);
                }

                let (logits, next_state) = model.forward(&input_batch, current_state)?;
                current_state = Some(next_state);

                // Optimization
                let logits_flat = logits.reshape((current_batch_size * seq_length, vocab_size))?;
                let target_flat = target_batch.reshape((current_batch_size * seq_length,))?; // IDs [N]

                // Cross Entropy
                // Candle cross_entropy expects logits and targets (u32 indices)
                let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
                
                total_loss += loss.to_scalar::<f32>()?;
                num_losses += 1;

                // Accuracy
                let preds = logits_flat.argmax(1)?;
                let correct_count = preds.eq(&target_flat)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
                correct += correct_count;
                total += current_batch_size * seq_length;

                let grads = loss.backward()?;
                optim_slstm.step(&grads)?;
                optim_mlstm.step(&grads)?;
                optim_other.step(&grads)?;

                if batch_idx % 10 == 0 || batch_idx == num_batches - 1 {
                    print!("\r  -> Batch [{}/{}] Loss: {:.4} Acc: {:.2}%", 
                        batch_idx + 1, num_batches, total_loss / (num_losses as f32),
                        100.0 * correct as f32 / total as f32);
                    io::stdout().flush().unwrap();
                }
            }
            println!();

            let avg_loss = total_loss / num_losses as f32;
            let accuracy = 100.0 * correct as f32 / total as f32;

            println!("Epoch [{:3}/{}], Loss: {:.4}, Accuracy: {:.2}%", epoch + 1, num_epochs, avg_loss, accuracy);

            // Save per epoch
            varmap.save(model_path)?;

            // Generate sample
             if epoch % 1 == 0 {
                let mut rng = rand::rng();
                let start_random = if tokens.len() > 10 {
                    rng.random_range(0..tokens.len() - 6)
                } else { 0 };
                
                let seed_tokens: Vec<usize> = tokens[start_random..start_random + 5].to_vec();
                let seed = tokenizer.decode(&seed_tokens);
                
                println!("  -> Generando con semilla al azar: '{}'", seed);
                let generated = generate_text(&model, &tokenizer, &seed, 100, vocab_size, &device)?;
                println!("  Generado: {}\n", generated);
             }
        }
        println!("\n¡Entrenamiento completado!");
    } else {
        // Just inference mode
    }

    // Modo interactivo
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO - GENERACIÓN DE TEXTO         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    loop {
        print!("Semilla > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }

         let seed = if input.eq_ignore_ascii_case("auto") {
             // We need 'text' but it might not be loaded if we didn't train.
             // If not trained, we might fail here. But simpler to assume we have text or don't support auto.
             if tokens.len() > 20 {
                 let seed_tokens: Vec<usize> = tokens[0..20].to_vec();
                 tokenizer.decode(&seed_tokens)
             } else {
                 "Once upon a time".to_string()
             }
        } else {
            input.to_string()
        };

        println!("\nGenerando...");
        let generated = generate_text(&model, &tokenizer, &seed, 200, vocab_size, &device)?;
        println!("Generado: {}\n", generated);
    }

    Ok(())
}
