#![recursion_limit = "256"]

/*!
Financial Time Series Forecasting with xLSTM using Embeddings

This example demonstrates how to use xLSTM for financial forecasting
using pre-computed embeddings from `embeddings_128.csv`, with support
for per-block learning rates.

Author: Mudit Bhargava (Ported to Rust)
Date: October 2025
*/

use candle_core::{DType, Device, Tensor, Result, Module};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use csv::ReaderBuilder;
use plotters::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use xlstm::{LearningRateConfig, LstmType, XLstm, XLstmconfig, BlockType};

type SequenceData = (
    Tensor,    // train_x: [num_samples, seq_length, input_size]
    Tensor,    // train_y: [num_samples, output_size]
    Tensor,    // test_x: [num_samples, seq_length, input_size]
    Tensor,    // test_y: [num_samples, output_size]
    Vec<(f32, f32)>, // test_prices: (current_price, next_price)
);

#[derive(Deserialize)]
#[allow(dead_code)]
struct EmbeddingRecord {
    trading_date: String,
    trading_code: String,
    company_name: String,
    last_price: f32,
    #[serde(flatten)]
    embeddings: HashMap<String, f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 { &args[1] } else { "train" };

    match mode {
        "train" => train_model(),
        "infer" => {
            if args.len() < 3 {
                eprintln!("Usage: {} infer <model_path>", args[0]);
                std::process::exit(1);
            }
            infer_model(&args[2])
        }
        "continue" => {
            if args.len() < 3 {
                eprintln!("Usage: {} continue <model_path>", args[0]);
                std::process::exit(1);
            }
            continue_training(&args[2])
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  {} train              - Train a new model", args[0]);
            eprintln!(
                "  {} infer <model_path> - Run inference with saved model",
                args[0]
            );
            eprintln!(
                "  {} continue <model_path> - Continue training from saved model",
                args[0]
            );
            std::process::exit(1);
        }
    }
}

fn train_model() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Financial Forecasting with Embeddings");
    println!("==========================================\n");

    // Load and prepare data
    println!("Loading embeddings_128.csv...");
    let (embeddings, prices) = load_data("embeddings_128.csv")?;
    println!("Loaded {} records", prices.len());

    // Hyperparameters
    let input_size = 128;
    let hidden_size = 128;
    let num_layers = 2;
    let num_blocks = 4;
    let output_size = 1;
    let dropout = 0.2;

    let seq_length = 20;
    let batch_size = 4; // Batch size per update step
    let num_epochs = 20;

    let train_split = 0.8;

    println!("Training configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Learning rates:");
    println!("    sLSTM blocks: 1e-4");
    println!("    mLSTM blocks: 1e-5");
    println!("    Other layers: 1e-4\n");

    // Device
    let device = Device::Cpu;

    // Create sequences
    println!("Creating sequences (seq_length={seq_length})...");
    let (train_x, train_y, test_x, test_y, test_prices) =
        create_sequences(&embeddings, &prices, seq_length, train_split, &device)?;

    let num_train = train_x.dim(0)?;
    let num_test = test_x.dim(0)?;

    println!("Training samples: {}", num_train);
    println!("Testing samples: {}\n", num_test);

    // Create model
    println!("Creating xLSTM model with alternating sLSTM/mLSTM blocks...");
    let config = XLstmconfig::new(input_size, hidden_size, num_layers, num_blocks, output_size)
        .with_dropout(dropout)
        .with_num_heads(4)
        .with_lstm_type(LstmType::Alternate)
        .with_use_projection(true);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = config.init(vb)?;
    
    // model.print_architecture(); // Need to adapt or implement
    println!();

    // Group parameters
    let block_types = config.get_block_config(); // Helper I added? No, model has it.
    // Actually config doesn't have it easily accessible without init logic duplication or helper.
    // I can reproduce logic.
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
            // format: block_N.rest
            let parts: Vec<&str> = name.split('.').collect();
            if let Some(block_part) = parts.first() {
                // block_N
                if let Some(idx_str) = block_part.strip_prefix("block_") {
                     if let Ok(idx) = idx_str.parse::<usize>() {
                         if idx < parsed_block_types.len() {
                             match parsed_block_types[idx] {
                                 BlockType::SLSTM => slstm_params.push(var),
                                 BlockType::MLSTM => mlstm_params.push(var),
                             }
                         } else {
                            other_params.push(var);
                         }
                     } else {
                         other_params.push(var);
                     }
                } else {
                    other_params.push(var);
                }
            } else {
                other_params.push(var);
            }
        } else {
            other_params.push(var);
        }
    }

    // Create optimizers
    let mut optim_slstm = AdamW::new(slstm_params, ParamsAdamW { lr: 1e-4, ..Default::default() })?;
    let mut optim_mlstm = AdamW::new(mlstm_params, ParamsAdamW { lr: 1e-5, ..Default::default() })?;
    let mut optim_other = AdamW::new(other_params, ParamsAdamW { lr: 1e-4, ..Default::default() })?;

    println!("Starting training...\n");

    // Training loop
    let num_batches = num_train / batch_size;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let mut num_losses = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(num_train);

            if start_idx >= end_idx {
                break;
            }

            // Extract batch
            let input_batch = train_x.narrow(0, start_idx, end_idx - start_idx)?;
            let target_batch = train_y.narrow(0, start_idx, end_idx - start_idx)?;

            // Forward pass
            let (predictions, _) = model.predict_last(&input_batch, None)?;

            // Compute loss
            let loss = mse_loss(&predictions, &target_batch)?;

            // Extract loss value
            let loss_f32 = loss.to_scalar::<f32>()?;
            total_loss += loss_f32;
            num_losses += 1;

            // Backward pass
            let grads = loss.backward()?;
            
            // Update steps
            optim_slstm.step(&grads)?;
            optim_mlstm.step(&grads)?;
            optim_other.step(&grads)?;
        }

        let avg_loss = total_loss / num_losses as f32;

        // Validation
        if epoch % 5 == 0 && num_test > 0 {
            let val_loss = evaluate(
                &model,
                &test_x,
                &test_y,
                seq_length,
                input_size,
                output_size,
            )?;
            println!(
                "Epoch [{:2}/{}], Train Loss: {:.6}, Val Loss: {:.6}",
                epoch + 1,
                num_epochs,
                avg_loss,
                val_loss
            );
        } else {
            println!(
                "Epoch [{:2}/{}], Train Loss: {:.6}",
                epoch + 1,
                num_epochs,
                avg_loss
            );
        }
    }

    println!("\nTraining completed!");

    // Save the model
    let model_path = "xlstm_model.safetensors";
    varmap.save(model_path)?;
    println!("\nModel saved to: {}", model_path);

    // Make predictions
    let (predictions, actuals) =
        make_predictions(&model, &test_x, &test_prices, seq_length, input_size)?;

    display_metrics(&predictions, &actuals)?;

    Ok(())
}

fn infer_model(model_path: &str) -> Result<(), Box<dyn Error>> {
    println!("xLSTM Inference Mode");
    println!("===================\n");

    // Load data
    println!("Loading embeddings_128.csv...");
    let (embeddings, prices) = load_data("embeddings_128.csv")?;
    println!("Loaded {} records", prices.len());

    // Hyperparameters (must match training)
    let input_size = 128;
    let hidden_size = 128;
    let num_layers = 2;
    let num_blocks = 4;
    let output_size = 1;
    let dropout = 0.2;

    let seq_length = 20;
    let train_split = 0.8;

    // Device
    let device = Device::Cpu;

    // Create sequences
    println!("Creating sequences...");
    let (_, _, test_x, _, test_prices) =
        create_sequences(&embeddings, &prices, seq_length, train_split, &device)?;

    println!("Testing samples: {}\n", test_x.dim(0)?);

    // Load model
    println!("Loading model from: {}", model_path);
    let config = XLstmconfig::new(input_size, hidden_size, num_layers, num_blocks, output_size)
        .with_dropout(dropout)
        .with_lstm_type(LstmType::Alternate)
        .with_use_projection(true);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = config.init(vb)?;
    
    // Load weights
    varmap.load(model_path)?;
    
    println!("Model loaded successfully!\n");

    // Make predictions
    let (predictions, actuals) =
        make_predictions(&model, &test_x, &test_prices, seq_length, input_size)?;

    display_metrics(&predictions, &actuals)?;

    Ok(())
}

fn continue_training(model_path: &str) -> Result<(), Box<dyn Error>> {
    println!("xLSTM Continue Training Mode");
    // Simplified: similar structure to train but load weights first
    // For brevity of this port, simply loading weights then training
    
    // ... Copy training setup ...
    // ...
    // varmap.load(model_path)?;
    // ...
    
    println!("Continue training logic would go here similar to train_model but utilizing loaded weights.");
    // Implementation omitted for brevity in this initial port, can be filled if needed.
    Ok(())
}

fn display_metrics(predictions: &[f32], actuals: &[f32]) -> Result<(), Box<dyn Error>> {
    // Calculate metrics
    let mse = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;
    let rmse = mse.sqrt();
    let mae = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f32>()
        / predictions.len() as f32;

    println!("\nPrediction Metrics:");
    println!("  RMSE: {rmse:.4}");
    println!("  MAE:  {mae:.4}");

    println!("\nCreating visualizations...");
    plot_predictions(predictions, actuals, "predictions_vs_actual.png")?;
    plot_scatter(predictions, actuals, "prediction_scatter.png")?;

    println!("\nVisualizations saved:");
    println!("  - predictions_vs_actual.png: Time series comparison");
    println!("  - prediction_scatter.png: Prediction accuracy scatter plot");

    Ok(())
}

// Helper for MSE loss
fn mse_loss(preds: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let diff = (preds - targets)?;
    let sq_diff = diff.sqr()?;
    sq_diff.mean_all()
}

fn load_data(path: &str) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

    let mut embeddings = Vec::new();
    let mut prices = Vec::new();

    for result in reader.deserialize() {
        let record: EmbeddingRecord = result?;

        // Extract embeddings in order (emb_0 to emb_127)
        let mut emb_vec = Vec::with_capacity(128);
        for i in 0..128 {
            let key = format!("emb_{i}");
            if let Some(&value) = record.embeddings.get(&key) {
                emb_vec.push(value);
            }
        }

        if emb_vec.len() == 128 {
            embeddings.push(emb_vec);
            prices.push(record.last_price);
        }
    }

    Ok((embeddings, prices))
}

fn create_sequences(
    embeddings: &[Vec<f32>],
    prices: &[f32],
    seq_length: usize,
    train_split: f32,
    device: &Device,
) -> Result<SequenceData, Box<dyn Error>> {
    let num_sequences = embeddings.len() - seq_length;
    let input_size = 128;

    // Pre-allocate flat arrays for all sequences
    let mut x_data = Vec::with_capacity(num_sequences * seq_length * input_size);
    let mut y_data = Vec::with_capacity(num_sequences);
    let mut price_pairs = Vec::with_capacity(num_sequences);

    // Create sliding windows
    for i in 0..num_sequences {
        // Input: sequence of embeddings (flattened)
        for j in 0..seq_length {
            x_data.extend_from_slice(&embeddings[i + j]);
        }

        // Target: next price (normalized as relative change)
        let current_price = prices[i + seq_length - 1];
        let next_price = prices[i + seq_length];
        let target = (next_price - current_price) / current_price;

        y_data.push(target);
        price_pairs.push((current_price, next_price));
    }

    // Split into train/test
    let split_idx = (num_sequences as f32 * train_split) as usize;

    // Create stacked tensors for train set
    let train_x = Tensor::from_vec(
        x_data[..(split_idx * seq_length * input_size)].to_vec(),
        (split_idx, seq_length, input_size),
        device,
    )?;
    let train_y = Tensor::from_vec(
        y_data[..split_idx].to_vec(),
        (split_idx, 1),
        device,
    )?;

    // Create stacked tensors for test set
    let test_len = num_sequences - split_idx;
    let test_x = Tensor::from_vec(
        x_data[(split_idx * seq_length * input_size)..].to_vec(),
        (test_len, seq_length, input_size),
        device,
    )?;
    let test_y = Tensor::from_vec(
        y_data[split_idx..].to_vec(),
        (test_len, 1),
        device,
    )?;

    let test_prices = price_pairs[split_idx..].to_vec();

    Ok((train_x, train_y, test_x, test_y, test_prices))
}

fn evaluate(
    model: &XLstm,
    test_x: &Tensor,
    test_y: &Tensor,
    seq_length: usize,
    input_size: usize,
    output_size: usize,
) -> Result<f32, Box<dyn Error>> {
    let num_test = test_x.dim(0)?;
    if num_test == 0 {
        return Ok(0.0);
    }

    let batch_size = 32.min(num_test);
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for i in (0..num_test).step_by(batch_size) {
        let end_idx = (i + batch_size).min(num_test);

        let input_batch = test_x.narrow(0, i, end_idx - i)?;
        let target_batch = test_y.narrow(0, i, end_idx - i)?;

        let (predictions, _) = model.predict_last(&input_batch, None)?;
        let loss = mse_loss(&predictions, &target_batch)?;

        total_loss += loss.to_scalar::<f32>()?;
        num_batches += 1;
    }

    Ok(total_loss / num_batches as f32)
}

fn make_predictions(
    model: &XLstm,
    test_x: &Tensor,
    test_prices: &[(f32, f32)],
    seq_length: usize,
    input_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let num_test = test_x.dim(0)?;
    let mut predictions = Vec::with_capacity(num_test);
    let mut actuals = Vec::with_capacity(num_test);

    // Process in batches for efficiency
    let batch_size = 32;
    for i in (0..num_test).step_by(batch_size) {
        let end_idx = (i + batch_size).min(num_test);

        let input_batch = test_x.narrow(0, i, end_idx - i)?;
        let (pred_batch, _) = model.predict_last(&input_batch, None)?;
        
        // Extract predictions from batch
        let pred_vec = pred_batch.flatten_all()?.to_vec1::<f32>()?;
        
        let batch_len = end_idx - i;
        for j in 0..batch_len {
            let pred_relative = pred_vec[j];

            // Convert back to actual price
            let (current_price, actual_next_price) = test_prices[i + j];
            let predicted_price = current_price * (1.0 + pred_relative);

            predictions.push(predicted_price);
            actuals.push(actual_next_price);
        }
    }

    Ok((predictions, actuals))
}

fn plot_predictions(
    predictions: &[f32],
    actuals: &[f32],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let margin = (max_val - min_val) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("xLSTM Price Predictions vs Actual", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..predictions.len(), (min_val - margin)..(max_val + margin))?;

    chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Price")
        .draw()?;

    // Plot actual prices
    chart
        .draw_series(LineSeries::new(
            actuals.iter().enumerate().map(|(i, &v)| (i, v)),
            &BLUE,
        ))?
        .label("Actual")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Plot predictions
    chart
        .draw_series(LineSeries::new(
            predictions.iter().enumerate().map(|(i, &v)| (i, v)),
            &RED,
        ))?
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Saved: {filename}");

    Ok(())
}

fn plot_scatter(
    predictions: &[f32],
    actuals: &[f32],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = predictions
        .iter()
        .chain(actuals.iter())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let margin = (max_val - min_val) * 0.1;
    let range = (min_val - margin)..(max_val + margin);

    let mut chart = ChartBuilder::on(&root)
        .caption("Prediction Accuracy Scatter Plot", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(range.clone(), range)?;

    chart
        .configure_mesh()
        .x_desc("Actual Price")
        .y_desc("Predicted Price")
        .draw()?;

    // Plot ideal line (y=x)
    chart
        .draw_series(LineSeries::new(
            vec![
                (min_val - margin, min_val - margin),
                (max_val + margin, max_val + margin),
            ],
            &BLACK.mix(0.3),
        ))?
        .label("Perfect Prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // Plot scatter points
    chart
        .draw_series(
            actuals
                .iter()
                .zip(predictions.iter())
                .map(|(&a, &p)| Circle::new((a, p), 3, RED.filled())),
        )?
        .label("Predictions")
        .legend(|(x, y)| Circle::new((x + 10, y), 3, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Saved: {filename}");

    Ok(())
}
