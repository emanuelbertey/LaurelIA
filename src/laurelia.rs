#![recursion_limit = "256"]

/*!
# LaurelIA: The Brain of xLSTM
Interactive CLI for Model Design, Training, and Generation.
*/

use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW, SGD};
use anyhow::Result;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::collections::HashSet;
use std::time::Instant;

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::AddedToken;

use xlstm::{LstmType, XLstm, XLstmconfig, BlockType};
use rand::Rng;

// --- ESTRUCTURAS DE APOYO ---

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self> {
        let special_tokens_strings = vec![
            "[ENG]".to_string(), "[SEP]".to_string(), "[ESP]".to_string(),
            "[EOS]".to_string(), "<PAD>".to_string(),
        ];

        let special_tokens: Vec<AddedToken> = special_tokens_strings
            .iter()
            .map(|t| AddedToken::from(t, true))
            .collect();

        let model = BPE::builder().byte_fallback(true).build().map_err(|e| anyhow::anyhow!(e))?;
        let mut tokenizer = HFTokenizer::new(model);

        tokenizer.with_pre_tokenizer(Some(Metaspace::new(' ', PrependScheme::Always, true)));

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
        let temp_file = "temp_train_laurelia.txt";
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
        self.tokenizer.save(path, true).map_err(|e| anyhow::anyhow!("Error al guardar: {}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Error al cargar: {}", e))?;
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

fn sample_from_logits(logits: &Tensor, temperature: f32) -> Result<usize> {
    let logits = logits.squeeze(0)?;
    let vocab_size = logits.dim(0)?;
    let scaled_logits = (&logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut probs_indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = 10;
    let top_k_probs = &probs_indexed[..k.min(vocab_size)];
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng(); 
    let mut sample: f32 = rng.random::<f32>() * sum;
    for (i, &p) in weights.iter().enumerate() {
        if sample <= p { return Ok(indices[i]); }
        sample -= p;
    }
    Ok(indices[0])
}

fn generate_text(
    model: &XLstm,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &Device,
) -> Result<String> {
    let mut current_text = seed_text.to_string();
    let seed_tokens = tokenizer.encode(seed_text);
    if seed_tokens.is_empty() { return Ok(current_text); }
    let mut current_state = None; 
    let mut current_tokens = seed_tokens;
    for i in 0..length {
        let tokens_to_process = if i == 0 { current_tokens.clone() } else { vec![*current_tokens.last().unwrap()] };
        let seq_len = tokens_to_process.len();
        let indices_vec: Vec<u32> = tokens_to_process.iter().map(|&t| t as u32).collect();
        let input = Tensor::from_vec(indices_vec, (1, seq_len), device)?;
        let (output, next_state) = model.forward(&input, current_state)?;
        current_state = Some(next_state.into_iter().map(|s| s.map(|state| state.detach())).collect());
        let last_logits = output.narrow(1, seq_len - 1, 1)?.squeeze(1)?.detach();
        let next_token = sample_from_logits(&last_logits, 0.8)?;
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

// --- CONTEXTO DE LAURELIA ---

struct LaurelState {
    config: Option<XLstmconfig>,
    model: Option<XLstm>,
    varmap: VarMap,
    tokenizer: Option<Tokenizer>,
    device: Device,
    
    // Stats persistentes (se guardan en el safetensors)
    stat_epoch: Var,
    stat_loss: Var,
    stat_acc: Var,

    // Hyper-params
    lr_mlstm: f64,
    lr_slstm: f64,
    lr_other: f64,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    epochs: usize,
    train_blocks: Vec<usize>,
    
    // Paths
    model_path: String,
    tokenizer_path: String,
    dataset_path: String,
}

impl LaurelState {
    fn new() -> Result<Self> {
        let varmap = VarMap::new();
        let device = Device::Cpu;
        
        // Inicializamos las Var de estadísticas
        let stat_epoch = Var::from_tensor(&Tensor::new(&[0.0f32], &device)?)?;
        let stat_loss = Var::from_tensor(&Tensor::new(&[0.0f32], &device)?)?;
        let stat_acc = Var::from_tensor(&Tensor::new(&[0.0f32], &device)?)?;

        // Las registramos en el varmap inicial
        {
            let mut data = varmap.data().lock().unwrap();
            data.insert("stats.epoch".to_string(), stat_epoch.clone());
            data.insert("stats.loss".to_string(), stat_loss.clone());
            data.insert("stats.acc".to_string(), stat_acc.clone());
        }

        Ok(Self {
            config: None,
            model: None,
            varmap,
            tokenizer: None,
            device,
            stat_epoch,
            stat_loss,
            stat_acc,
            lr_mlstm: 4e-5,
            lr_slstm: 2e-4,
            lr_other: 2e-4,
            batch_size: 16,
            seq_length: 128,
            stride: 64,
            epochs: 10,
            train_blocks: vec![],
            model_path: "laurelia_model.safetensors".to_string(),
            tokenizer_path: "laurelia_tokenizer.json".to_string(),
            dataset_path: "input.txt".to_string(),
        })
    }

    fn print_status(&self) {
        println!("\n--- LaurelIA STATUS ---");
        let cur_epoch = self.stat_epoch.as_tensor().to_vec1::<f32>().unwrap_or(vec![0.0])[0];
        let cur_loss = self.stat_loss.as_tensor().to_vec1::<f32>().unwrap_or(vec![0.0])[0];
        let cur_acc = self.stat_acc.as_tensor().to_vec1::<f32>().unwrap_or(vec![0.0])[0];

        println!("  Estado Entrenamiento: Época: {:.0}, Loss: {:.4}, Acc: {:.2}%", cur_epoch, cur_loss, cur_acc);
        println!("  Modelo:      {}", if self.model.is_some() { "LISTO" } else { "No inicializado" });
        println!("  Tokenizador: {}", if self.tokenizer.is_some() { "LISTO" } else { "No inicializado" });
        println!("  Dataset:     {}", self.dataset_path);
        println!("  Hidden Size: {}", self.config.as_ref().map(|c| c.hidden_size).unwrap_or(0));
        println!("  LRs:         mLSTM: {:.1e}, sLSTM: {:.1e}", self.lr_mlstm, self.lr_slstm);
        println!("-----------------------\n");
    }
}

fn main() -> Result<()> {
    println!("\n\n");
    println!("   ██╗      █████╗ ██╗   ██╗██████╗ ███████╗██╗     ██╗ █████╗ ");
    println!("   ██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝██║     ██║██╔══██╗");
    println!("   ██║     ███████║██║   ██║██████╔╝█████╗  ██║     ██║███████║");
    println!("   ██║     ██╔══██╗██║   ██║██╔══██╗██╔══╝  ██║     ██║██╔══██╗");
    println!("   ███████╗██║  ██║╚██████╔╝██║  ██║███████╗███████╗██║██║  ██║");
    println!("   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝");
    println!("                  The Brain of xLSTM Design\n");

    let mut state = LaurelState::new()?;

    loop {
        print!("LaurelIA > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let parts: Vec<&str> = input.trim().split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0].to_lowercase().as_str() {
            "help" => print_help(),
            "status" => state.print_status(),
            "create" => interactive_create(&mut state)?,
            "train" => run_training(&mut state)?,
            "generate" | "gen" => run_generation(&mut state)?,
            "config" => update_config(&mut state, &parts)?,
            "load" => load_all(&mut state, &parts)?,
            "save" => save_model(&state)?,
            "tokenizer" => handle_tokenizer(&mut state, &parts)?,
            "exit" | "salir" => break,
            _ => {
                // Intento de carga automática para cualquier cosa que parezca un archivo o comando de carga
                if parts[0].contains('.') || parts.len() > 1 {
                    auto_load(&mut state, &parts)?;
                } else {
                    println!("Comando desconocido. Escribe 'help' para ver la lista.");
                }
            }
        }
    }

    Ok(())
}

fn print_help() {
    println!("\n╔══════════════════════════ HELP ══════════════════════════╗");
    println!("║ create     - Diseña un nuevo modelo interactivamente     ║");
    println!("║ train      - Inicia el entrenamiento con dataset actual  ║");
    println!("║ status     - Muestra el estado actual de LaurelIA        ║");
    println!("║ generate   - Entra en modo chat/generación               ║");
    println!("║ load       - Carga modelo [path] y/o tokenizer [path]    ║");
    println!("║ save       - Guarda el estado actual del modelo          ║");
    println!("║ config     - Ajusta parámetros (lr, batch, seq, blocks)  ║");
    println!("║ tokenizer  - Entrena o carga un tokenizer específico     ║");
    println!("║ exit       - Salir de LaurelIA                           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

fn interactive_create(state: &mut LaurelState) -> Result<()> {
    println!("\n--- DISEÑO DE ARQUITECTURA LaurelIA ---");
    
    print!("Tamaño Hidden (ej. 512): "); io::stdout().flush()?;
    let mut h = String::new(); io::stdin().read_line(&mut h)?;
    let hidden = h.trim().parse::<usize>().unwrap_or(512);

    print!("Número de Bloques (ej. 4): "); io::stdout().flush()?;
    let mut b = String::new(); io::stdin().read_line(&mut b)?;
    let blocks = b.trim().parse::<usize>().unwrap_or(4);

    print!("Heads para mLSTM (ej. 4): "); io::stdout().flush()?;
    let mut hd = String::new(); io::stdin().read_line(&mut hd)?;
    let heads = hd.trim().parse::<usize>().unwrap_or(4);

    println!("Tipo de LSTM: 1. mLSTM (Todo), 2. sLSTM (Todo), 3. Alternate");
    print!("Opción: "); io::stdout().flush()?;
    let mut opt = String::new(); io::stdin().read_line(&mut opt)?;
    let l_type = match opt.trim() {
        "1" => LstmType::MLSTM,
        "2" => LstmType::SLSTM,
        _ => LstmType::Alternate,
    };

    let vocab = state.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(1024);
    
    let config = XLstmconfig::new(hidden, hidden, 1, blocks, vocab)
        .with_vocab_size(vocab)
        .with_num_heads(heads)
        .with_lstm_type(l_type)
        .with_use_projection(true)
        .with_dropout(0.05);

    state.config = Some(config.clone());
    state.varmap = VarMap::new();
    
    // IMPORTANTE: Re-registrar las Var de estadísticas en el NUEVO VarMap
    {
        let mut data = state.varmap.data().lock().unwrap();
        data.insert("stats.epoch".to_string(), state.stat_epoch.clone());
        data.insert("stats.loss".to_string(), state.stat_loss.clone());
        data.insert("stats.acc".to_string(), state.stat_acc.clone());
    }

    let vb = VarBuilder::from_varmap(&state.varmap, DType::F32, &state.device);
    state.model = Some(config.init(vb)?);

    let total_params: usize = state.varmap.data().lock().unwrap().values().map(|v| v.as_tensor().elem_count()).sum();
    println!("\nModelo creado con éxito. {:.2}M parámetros.", total_params as f32 / 1_000_000.0);
    Ok(())
}

fn update_config(state: &mut LaurelState, parts: &[&str]) -> Result<()> {
    if parts.len() < 3 {
        println!("Uso: config <lr|batch|seq|blocks> <valor>");
        return Ok(());
    }

    match parts[1] {
        "lr_mlstm" => state.lr_mlstm = parts[2].parse()?,
        "lr_slstm" => state.lr_slstm = parts[2].parse()?,
        "lr_other" => state.lr_other = parts[2].parse()?,
        "batch" => state.batch_size = parts[2].parse()?,
        "seq" => state.seq_length = parts[2].parse()?,
        "stride" => state.stride = parts[2].parse()?,
        "epochs" => state.epochs = parts[2].parse()?,
        "blocks" => {
            state.train_blocks = parts[2].split(',').filter_map(|s| s.parse().ok()).collect();
        }
        _ => println!("Configuración no reconocida."),
    }
    println!("Configuración actualizada.");
    Ok(())
}

fn run_training(state: &mut LaurelState) -> Result<()> {
    if state.model.is_none() || state.tokenizer.is_none() {
        println!("ERROR: Debes tener un modelo y un tokenizador listos. Usa 'create' y 'tokenizer'.");
        return Ok(());
    }

    println!("\nCargando dataset desde {}...", state.dataset_path);
    let text = fs::read_to_string(&state.dataset_path)?;
    let tokens = state.tokenizer.as_ref().unwrap().encode(&text);
    println!("Tokens totales: {}", tokens.len());

    let num_sequences = tokens.len().saturating_sub(state.seq_length);
    let num_actual_sequences = (num_sequences + state.stride - 1) / state.stride;
    let num_batches = num_actual_sequences.div_ceil(state.batch_size);

    // Organizar parámetros
    let mut slstm_params = Vec::new();
    let mut mlstm_params = Vec::new();
    let mut other_params = Vec::new();

    let config = state.config.as_ref().unwrap();
    let block_types = match &config.lstm_type {
        LstmType::SLSTM => vec![BlockType::SLSTM; config.num_blocks],
        LstmType::MLSTM => vec![BlockType::MLSTM; config.num_blocks],
        _ => (0..config.num_blocks).map(|i| if i % 2 == 0 { BlockType::SLSTM } else { BlockType::MLSTM }).collect(),
    };

    let data = state.varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if name.starts_with("block_") {
            let parts: Vec<&str> = name.split('.').collect();
            if let Some(idx_str) = parts[0].strip_prefix("block_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if !state.train_blocks.is_empty() && !state.train_blocks.contains(&idx) { continue; }
                    if idx < block_types.len() {
                        match block_types[idx] {
                            BlockType::SLSTM => slstm_params.push(var.clone()),
                            BlockType::MLSTM => mlstm_params.push(var.clone()),
                        }
                    } else { other_params.push(var.clone()); }
                }
            }
        } else { other_params.push(var.clone()); }
    }
    drop(data);

    let mut opt_mlstm = if !mlstm_params.is_empty() { Some(AdamW::new(mlstm_params.clone(), ParamsAdamW { lr: state.lr_mlstm, ..Default::default() })?) } else { None };
    let mut opt_slstm = if !slstm_params.is_empty() { Some(SGD::new(slstm_params.clone(), state.lr_slstm)?) } else { None };
    let mut opt_other = if !other_params.is_empty() { Some(SGD::new(other_params.clone(), state.lr_other)?) } else { None };

    let active_params: Vec<Var> = [slstm_params, mlstm_params, other_params].concat();

    println!("Entrenamiento iniciado: {} batches x {} épocas", num_batches, state.epochs);

    for epoch in 0..state.epochs {
        let start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut num_losses = 0;
        let mut correct = 0;
        let mut total = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * state.batch_size;
            let current_size = (state.batch_size).min(num_actual_sequences - start_idx);
            if current_size < state.batch_size { break; }

            let (x, y) = create_batch(&tokens, start_idx * state.stride, current_size, state.seq_length, state.stride, &state.device)?;
            let (logits, _) = state.model.as_ref().unwrap().forward(&x, None)?;
            
            let vocab_size = state.tokenizer.as_ref().unwrap().vocab_size();
            let logits_flat = logits.reshape((current_size * state.seq_length, vocab_size))?;
            let target_flat = y.reshape((current_size * state.seq_length,))?;
            
            let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
            total_loss += loss.to_scalar::<f32>()?;
            num_losses += 1;
            
            let preds = logits_flat.argmax(1)?;
            correct += preds.eq(&target_flat)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
            total += current_size * state.seq_length;

            let grads = loss.backward()?;
            
            // Clipping
            let mut gn = 0.0f32;
            for v in active_params.iter() {
                if let Some(g) = grads.get(v) { gn += g.sqr()?.sum_all()?.to_scalar::<f32>()?; }
            }
            let clip = (1.0 / (gn.sqrt() + 1e-6)).min(1.0);
            if clip < 1.0 {
                for v in active_params.iter() {
                    if let Some(g) = grads.get(v) { let _ = g.affine(clip as f64, 0.0)?; }
                }
            }

            if let Some(o) = &mut opt_mlstm { o.step(&grads)?; }
            if let Some(o) = &mut opt_slstm { o.step(&grads)?; }
            if let Some(o) = &mut opt_other { o.step(&grads)?; }
            
            if batch_idx % 10 == 0 {
                print!("\r  Epoch {} [{}/{}] Loss: {:.4} ({:.1}s)", epoch+1, batch_idx+1, num_batches, total_loss / (batch_idx+1) as f32, start.elapsed().as_secs_f32());
                io::stdout().flush()?;
            }
        }
        let avg_loss = total_loss / num_losses as f32;
        let accuracy = 100.0 * correct as f32 / total as f32;
        
        // Actualizamos estadísticas persistentes
        state.stat_epoch.set(&Tensor::new(&[epoch as f32 + 1.0], &state.device)?)?;
        state.stat_loss.set(&Tensor::new(&[avg_loss], &state.device)?)?;
        state.stat_acc.set(&Tensor::new(&[accuracy], &state.device)?)?;

        println!("\nEpoch {} completada. Acc: {:.2}%. Guardando modelo y stats...", epoch+1, accuracy);
        state.varmap.save(&state.model_path)?;
    }
    Ok(())
}

fn run_generation(state: &mut LaurelState) -> Result<()> {
    if state.model.is_none() || state.tokenizer.is_none() {
        println!("ERROR: Carga o crea un modelo y tokenizador primero.");
        return Ok(());
    }
    
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        LaurelIA - MODO INTERACTIVO DE CHAT            ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("Comandos:");
    println!("  - Escribe tu texto y presiona Enter para generar.");
    println!("  - 'len N' : Cambia la longitud a N tokens.");
    println!("  - 'auto'  : Usa una semilla automática del dataset.");
    println!("  - 'salir' : Regresa al menú principal.\n");

    let mut gen_length: usize = 100;

    loop {
        print!("Chat > "); 
        io::stdout().flush()?;
        let mut input = String::new(); 
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() { continue; }
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }

        if input.to_lowercase().starts_with("len") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(n) = parts[1].parse::<usize>() {
                    gen_length = n;
                    println!("Longitud establecida en {} tokens.\n", gen_length);
                    continue;
                }
            }
        }

        let seed = if input.eq_ignore_ascii_case("auto") {
            "El sistema xLSTM es".to_string() 
        } else {
            input.to_string()
        };

        println!("Pensando...");
        let generated = generate_text(
            state.model.as_ref().unwrap(), 
            state.tokenizer.as_ref().unwrap(), 
            &seed, 
            gen_length, 
            &state.device
        )?;
        println!("\nLaurelIA: {}\n", generated);
    }
    Ok(())
}

fn permissive_load(varmap: &VarMap, path: &str, device: &Device) -> Result<()> {
    // Cargamos los tensores del archivo safetensors de forma manual
    let tensors = candle_core::safetensors::load(path, device)?;
    let mut data = varmap.data().lock().unwrap();
    
    let mut missing_stats = Vec::new();
    let mut missing_params = Vec::new();

    for (name, var) in data.iter_mut() {
        if let Some(t) = tensors.get(name) {
            var.set(t)?;
        } else {
            if name.starts_with("stats.") {
                missing_stats.push(name.clone());
            } else {
                missing_params.push(name.clone());
            }
        }
    }

    if !missing_stats.is_empty() {
        println!("Nota: El modelo no contenía estadísticas ({:?}). Se iniciarán en 0.", missing_stats);
    }
    
    if !missing_params.is_empty() {
        println!("¡ADVERTENCIA! Faltan parámetros críticos del modelo: {}...", missing_params[0]);
        println!("Es probable que la arquitectura actual no coincida con el archivo cargado.");
        anyhow::bail!("Faltan parámetros críticos en el modelo.");
    }

    Ok(())
}

fn auto_load(state: &mut LaurelState, parts: &[&str]) -> Result<()> {
    let start_idx = if parts[0].to_lowercase() == "load" { 1 } else { 0 };
    if parts.len() <= start_idx { return Ok(()); }

    let file_parts = &parts[start_idx..];
    
    // --- 1. PRIMERO CARGAMOS LOS TOKENIZADORES (Prioridad) ---
    for path_raw in file_parts {
        if path_raw.ends_with(".json") || path_raw.to_lowercase().contains("tokenizer") {
            let path = if path_raw.ends_with(".json") { path_raw.to_string() } else { format!("{}.json", path_raw) };
            if let Ok(t) = Tokenizer::load(&path) {
                state.tokenizer = Some(t);
                state.tokenizer_path = path;
                println!("✓ Tokenizador cargado: {}", state.tokenizer_path);
            }
        }
    }

    // --- 2. LUEGO CARGAMOS LOS MODELOS ---
    for path_raw in file_parts {
        if path_raw.ends_with(".safetensors") || path_raw.to_lowercase().contains("model") {
            let path = if path_raw.ends_with(".safetensors") { path_raw.to_string() } else { format!("{}.safetensors", path_raw) };
            
            // --- ESCÁNER DE ADN xLSTM (Con Fallback Interactivo) ---
            if state.config.is_none() {
                println!("🕵 Analizando archivo '{}'...", path);
                let mut h_final = None;
                let mut v_final = state.tokenizer.as_ref().map(|t| t.vocab_size());
                let mut blocks_final = None;
                let mut sandwich = Vec::new();

                if let Ok(tensors) = candle_core::safetensors::load(&path, &state.device) {
                    let mut block_map = std::collections::BTreeMap::new();
                    
                    for (name, tensor) in tensors.iter() {
                        let dims = tensor.dims();
                        // Intentar deducir Hidden si no lo tenemos
                        if h_final.is_none() && (name.contains("embedding.weight") || name.contains("head.linear2.weight")) {
                            if dims[0] > dims[1] { h_final = Some(dims[1]); if v_final.is_none() { v_final = Some(dims[0]); } }
                            else { h_final = Some(dims[0]); if v_final.is_none() { v_final = Some(dims[1]); } }
                        }
                        // Mapear bloques
                        if let Some(pos) = name.find("block_") {
                            let rest = &name[pos + 6..];
                            if let Ok(idx) = rest.split('.').next().unwrap_or("").parse::<usize>() {
                                if name.contains(".mlstm.") { block_map.insert(idx, BlockType::MLSTM); }
                                else if name.contains(".slstm.") { block_map.insert(idx, BlockType::SLSTM); }
                            }
                        }
                    }
                    
                    if !block_map.is_empty() {
                        let max_idx = block_map.keys().max().unwrap();
                        for i in 0..=*max_idx {
                            sandwich.push(*block_map.get(&i).unwrap_or(&BlockType::MLSTM));
                        }
                        blocks_final = Some(sandwich.len());
                    }
                }

                // --- FALLBACK INTERACTIVO: Si falta algo, preguntamos ---
                if h_final.is_none() || blocks_final.is_none() {
                    println!("⚠ No pude deducir toda la arquitectura automáticamente.");
                    
                    if h_final.is_none() {
                        print!("   Ingresa Hidden Size (ej. 512): "); io::stdout().flush()?;
                        let mut input = String::new(); io::stdin().read_line(&mut input)?;
                        h_final = Some(input.trim().parse().unwrap_or(512));
                    }
                    if blocks_final.is_none() {
                        print!("   Ingresa Número de Bloques (ej. 1): "); io::stdout().flush()?;
                        let mut input = String::new(); io::stdin().read_line(&mut input)?;
                        let n = input.trim().parse().unwrap_or(1);
                        blocks_final = Some(n);
                        sandwich = vec![BlockType::MLSTM; n]; // Default a mLSTM
                    }
                }

                let h = h_final.unwrap_or(512);
                let v = v_final.unwrap_or(1024);
                let b = blocks_final.unwrap_or(1);

                println!("✓ Configurando: H={}, V={}, Bloques={}", h, v, b);
                let config = XLstmconfig::new(h, h, 1, b, v)
                    .with_vocab_size(v)
                    .with_num_heads(2) // Fallback para heads
                    .with_lstm_type(LstmType::Custom(sandwich))
                    .with_use_projection(true);
                
                state.config = Some(config.clone());
                let vb = VarBuilder::from_varmap(&state.varmap, DType::F32, &state.device);
                if let Ok(m) = config.init(vb) {
                    state.model = Some(m);
                }
            }

            match permissive_load(&state.varmap, &path, &state.device) {
                Ok(_) => { 
                    state.model_path = path;
                    println!("✓ Pesos inyectados.");
                    state.print_status();
                },
                Err(e) => println!("✗ Error al inyectar pesos: {}. Verifica los parámetros.", e),
            }
        }
    }
    Ok(())
}


fn load_all(state: &mut LaurelState, parts: &[&str]) -> Result<()> {
    if parts.len() < 2 {
        println!("Uso: load <archivo1> <archivo2> ...");
        return Ok(());
    }

    match parts[1].to_lowercase().as_str() {
        "all" => {
            if Path::new("laurelia_tokenizer.json").exists() {
                state.tokenizer = Some(Tokenizer::load("laurelia_tokenizer.json")?);
                println!("Tokenizador por defecto cargado.");
            }
            if Path::new("laurelia_model.safetensors").exists() && state.config.is_some() {
                permissive_load(&state.varmap, "laurelia_model.safetensors", &state.device)?;
                println!("Modelo por defecto cargado.");
            }
        }
        _ => return auto_load(state, parts),
    }
    Ok(())
}

fn save_model(state: &LaurelState) -> Result<()> {
    state.varmap.save(&state.model_path)?;
    println!("Modelo guardado en {}", state.model_path);
    Ok(())
}

fn handle_tokenizer(state: &mut LaurelState, parts: &[&str]) -> Result<()> {
    if parts.len() < 2 {
        println!("Uso: tokenizer <train|load> [vocab_size]");
        return Ok(());
    }
    if parts[1] == "train" {
        let vocab = parts.get(2).unwrap_or(&"1024").parse()?;
        let text = fs::read_to_string(&state.dataset_path)?;
        state.tokenizer = Some(Tokenizer::from_text(&text, vocab)?);
        state.tokenizer.as_ref().unwrap().save(&state.tokenizer_path)?;
        println!("Tokenizador entrenado y guardado.");
    } else {
        state.tokenizer = Some(Tokenizer::load(&state.tokenizer_path)?);
        println!("Tokenizador cargado.");
    }
    Ok(())
}
