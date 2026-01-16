/*!
# xLSTM: Extended Long Short-Term Memory Model

This module implements the main xLSTM model that can stack multiple blocks
with flexible mixing of sLSTM and mLSTM, including support for per-block
learning rates.

Author: Mudit Bhargava (Ported to Rust)
Date: October 2025
*/

use burn::{
    config::Config,
    module::{Module, ParamId},
    nn::{Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    optim::{GradientsParams, Optimizer},
    tensor::{
        Tensor, activation,
        backend::{AutodiffBackend, Backend},
    },
};
use num_traits::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};

use crate::{BlockType, XLstmblock, XLstmblockConfig, block::LSTMState};

/// Configuration for LSTM type in xLSTM model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum LstmType {
    /// All blocks use sLSTM
    SLSTM,
    /// All blocks use mLSTM
    MLSTM,
    /// Alternating pattern: sLSTM, mLSTM, sLSTM, mLSTM, ...
    Alternate,
    /// Custom pattern specified by user
    Custom(alloc::vec::Vec<BlockType>),
}

/// Learning rate configuration for different parts of the model
#[derive(Debug, Clone)]
pub enum LearningRateConfig {
    /// Uniform learning rate for all parameters
    Uniform(f64),
    /// Different learning rates for sLSTM vs mLSTM blocks
    PerBlockType {
        slstm_lr: f64,
        mlstm_lr: f64,
        other_lr: f64, // For input projection and output head
    },
    /// Explicit learning rate per block (length must match num_blocks)
    /// Also includes learning rate for input projection and output head
    PerBlock {
        block_lrs: alloc::vec::Vec<f64>,
        other_lr: f64,
    },
}

impl LearningRateConfig {
    /// Create a uniform learning rate configuration
    pub fn uniform(lr: f64) -> Self {
        Self::Uniform(lr)
    }

    /// Create per-block-type learning rate configuration
    pub fn per_block_type(slstm_lr: f64, mlstm_lr: f64, other_lr: f64) -> Self {
        Self::PerBlockType {
            slstm_lr,
            mlstm_lr,
            other_lr,
        }
    }

    /// Create per-block learning rate configuration
    pub fn per_block(block_lrs: alloc::vec::Vec<f64>, other_lr: f64) -> Self {
        Self::PerBlock {
            block_lrs,
            other_lr,
        }
    }
}

/// Configuration for xLSTM model
#[derive(Config, Debug)]
pub struct XLstmconfig {
    /// Input size (number of features)
    pub input_size: usize,
    /// Hidden size in LSTM blocks
    pub hidden_size: usize,
    /// Number of layers per block
    pub num_layers: usize,
    /// Number of blocks
    pub num_blocks: usize,
    /// Output size (for prediction)
    pub output_size: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Whether to use bidirectional LSTM
    #[config(default = "false")]
    pub bidirectional: bool,
    /// Number of heads for mLSTM blocks
    #[config(default = "4")]
    pub num_heads: usize,
    /// LSTM type configuration
    #[config(default = "LstmType::SLSTM")]
    pub lstm_type: LstmType,
    /// Whether to use input projection
    #[config(default = "true")]
    pub use_projection: bool,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:0.1}")]
    pub initializer: Initializer,
}

impl XLstmconfig {
    /// Initialize a new xLSTM model
    pub fn init<B: Backend>(&self, device: &B::Device) -> XLstm<B> {
        // Parse block types
        let block_types = self.parse_lstm_type();

        // Create input projection if requested
        let (input_projection, block_input_size) = if self.use_projection {
            let linear = LinearConfig::new(self.input_size, self.hidden_size).init(device);
            let norm = LayerNormConfig::new(self.hidden_size).init(device);
            let dropout = DropoutConfig::new(self.dropout).init();
            (Some((linear, norm, dropout)), self.hidden_size)
        } else {
            (None, self.input_size)
        };

        // Create blocks
        let blocks: alloc::vec::Vec<XLstmblock<B>> = block_types
            .iter()
            .map(|&block_type| {
                XLstmblockConfig::new(
                    block_input_size,
                    self.hidden_size,
                    self.num_layers,
                    block_type,
                )
                .with_num_heads(self.num_heads)
                .with_dropout(self.dropout)
                .with_initializer(self.initializer.clone())
                .init(device)
            })
            .collect();

        // Create output head
        let output_head = (
            LinearConfig::new(block_input_size, self.hidden_size).init(device),
            DropoutConfig::new(self.dropout).init(),
            LinearConfig::new(self.hidden_size, self.output_size).init(device),
        );

        XLstm {
            input_projection,
            blocks,
            output_head,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            output_size: self.output_size,
            num_blocks: self.num_blocks,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
            use_projection: self.use_projection,
        }
    }

    fn parse_lstm_type(&self) -> alloc::vec::Vec<BlockType> {
        match &self.lstm_type {
            LstmType::SLSTM => alloc::vec![BlockType::SLSTM; self.num_blocks],
            LstmType::MLSTM => alloc::vec![BlockType::MLSTM; self.num_blocks],
            LstmType::Alternate => (0..self.num_blocks)
                .map(|i| {
                    if i % 2 == 0 {
                        BlockType::SLSTM
                    } else {
                        BlockType::MLSTM
                    }
                })
                .collect(),
            LstmType::Custom(types) => {
                assert!(
                    (types.len() == self.num_blocks),
                    "Custom LSTM type length ({}) must match num_blocks ({})",
                    types.len(),
                    self.num_blocks
                );

                types.clone()
            }
        }
    }
}

/// Main xLSTM model for sequence processing
#[derive(Module, Debug)]
pub struct XLstm<B: Backend> {
    /// Optional input projection layers
    pub input_projection: Option<(Linear<B>, LayerNorm<B>, Dropout)>,
    /// Stack of xLSTM blocks
    pub blocks: alloc::vec::Vec<XLstmblock<B>>,
    /// Output head layers
    pub output_head: (Linear<B>, Dropout, Linear<B>),
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Output size
    pub output_size: usize,
    /// Number of blocks
    pub num_blocks: usize,
    /// Number of layers per block
    pub num_layers: usize,
    /// Number of heads (for mLSTM blocks)
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether input projection is used
    pub use_projection: bool,
}

impl<B: Backend> XLstm<B> {
    /// Forward pass through xLSTM model
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - Optional initial states for each block
    ///
    /// # Returns
    /// * Output tensor [`batch_size`, `seq_length`, `output_size`]
    /// * Final states for each block
    pub fn forward(
        &self,
        input_seq: Tensor<B, 3>,
        states: Option<alloc::vec::Vec<Option<LSTMState<B>>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<Option<LSTMState<B>>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive,
    {
        // Apply input projection if present
        let mut x = if let Some((linear, norm, dropout)) = &self.input_projection {
            let mut x = linear.forward(input_seq);
            x = norm.forward(x);
            x = activation::gelu(x);
            dropout.forward(x)
        } else {
            input_seq
        };

        // Initialize states if not provided
        let mut hidden_states = states.unwrap_or_else(|| alloc::vec![None; self.num_blocks]);

        // Pass through blocks
        for (i, block) in self.blocks.iter().enumerate() {
            let old_state = hidden_states[i].take();
            let (output, new_state) = block.forward(x, old_state);
            x = output;
            hidden_states[i] = new_state;
        }

        // Apply output head
        let (linear1, dropout, linear2) = &self.output_head;
        x = linear1.forward(x);
        x = activation::gelu(x);
        x = dropout.forward(x);
        let output = linear2.forward(x);

        (output, hidden_states)
    }

    /// Forward pass returning only the last timestep prediction
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - Optional initial states
    ///
    /// # Returns
    /// * Last timestep output [`batch_size`, `output_size`]
    /// * Final states
    pub fn predict_last(
        &self,
        input_seq: Tensor<B, 3>,
        states: Option<alloc::vec::Vec<Option<LSTMState<B>>>>,
    ) -> (Tensor<B, 2>, alloc::vec::Vec<Option<LSTMState<B>>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive,
    {
        let (output, states) = self.forward(input_seq, states);
        let [batch_size, seq_length, _] = output.dims();
        let last_output = output
            .slice([
                0..batch_size,
                (seq_length - 1)..seq_length,
                0..self.output_size,
            ])
            .squeeze(1);
        (last_output, states)
    }

    /// Get block configuration
    pub fn get_block_config(&self) -> alloc::vec::Vec<BlockType> {
        self.blocks
            .iter()
            .map(super::block::XLstmblock::get_type)
            .collect()
    }

    /// Print model architecture summary
    pub fn print_architecture(&self) {
        println!("xLSTM Model Architecture:");
        println!("  Input size: {}", self.input_size);
        println!("  Hidden size: {}", self.hidden_size);
        println!("  Output size: {}", self.output_size);
        println!("  Layers per block: {}", self.num_layers);
        println!("  Number of blocks: {}", self.num_blocks);
        println!("  Heads (mLSTM): {}", self.num_heads);
        println!("  Dropout: {}", self.dropout);
        println!("  Use input projection: {}", self.use_projection);
        println!("\nBlock Configuration:");
        for (i, block) in self.blocks.iter().enumerate() {
            let type_str = match block.get_type() {
                BlockType::SLSTM => "sLSTM",
                BlockType::MLSTM => "mLSTM",
            };
            println!("    Block {}: {}", i + 1, type_str);
        }
    }
}

// Extension trait for per-block learning rate optimization
impl<B: AutodiffBackend> XLstm<B> {
    /// Get parameter IDs for a specific block
    pub fn get_block_param_ids(&self, block_idx: usize) -> alloc::vec::Vec<ParamId> {
        if block_idx >= self.blocks.len() {
            return alloc::vec![];
        }

        use burn::module::list_param_ids;
        list_param_ids(&self.blocks[block_idx])
    }

    /// Get parameter IDs for input projection and output head
    pub fn get_other_param_ids(&self) -> alloc::vec::Vec<ParamId> {
        use burn::module::list_param_ids;

        let mut ids = alloc::vec::Vec::new();

        if let Some((linear, norm, _dropout)) = &self.input_projection {
            ids.extend(list_param_ids(linear));
            ids.extend(list_param_ids(norm));
        }

        let (linear1, _dropout, linear2) = &self.output_head;
        ids.extend(list_param_ids(linear1));
        ids.extend(list_param_ids(linear2));

        ids
    }

    /// Get all parameter IDs for the model
    pub fn get_all_param_ids(&self) -> alloc::vec::Vec<ParamId> {
        burn::module::list_param_ids(self)
    }

    /// Apply optimizer step with per-block learning rates
    pub fn optimizer_step<O: Optimizer<Self, B>>(
        self,
        lr_config: &LearningRateConfig,
        optimizer: &mut O,
        mut grads: B::Gradients,
    ) -> Self {
        match lr_config {
            LearningRateConfig::Uniform(lr) => {
                // Simple case: single learning rate for everything
                let grads_params = GradientsParams::from_grads(grads, &self);

                optimizer.step(*lr, self, grads_params)
            }
            LearningRateConfig::PerBlockType {
                slstm_lr,
                mlstm_lr,
                other_lr,
            } => {
                // Collect block types before we start moving model
                let block_info: alloc::vec::Vec<(usize, BlockType, f64)> = self
                    .blocks
                    .iter()
                    .enumerate()
                    .map(|(i, block)| {
                        let lr = match block.get_type() {
                            BlockType::SLSTM => *slstm_lr,
                            BlockType::MLSTM => *mlstm_lr,
                        };
                        (i, block.get_type(), lr)
                    })
                    .collect();

                let mut model = self;

                // Process each block
                for (i, _block_type, lr) in block_info {
                    let param_ids = model.get_block_param_ids(i);
                    let block_grads = GradientsParams::from_params(&mut grads, &model, &param_ids);
                    model = optimizer.step(lr, model, block_grads);
                }

                // Process input projection and output head
                let other_param_ids = model.get_other_param_ids();
                let other_grads =
                    GradientsParams::from_params(&mut grads, &model, &other_param_ids);
                model = optimizer.step(*other_lr, model, other_grads);

                model
            }
            LearningRateConfig::PerBlock {
                block_lrs,
                other_lr,
            } => {
                assert_eq!(
                    block_lrs.len(),
                    self.blocks.len(),
                    "block_lrs length must match number of blocks"
                );

                let num_blocks = self.blocks.len();
                let mut model = self;

                // Process each block with its specific learning rate
                for i in 0..num_blocks {
                    let lr = block_lrs[i];
                    let param_ids = model.get_block_param_ids(i);
                    let block_grads = GradientsParams::from_params(&mut grads, &model, &param_ids);
                    model = optimizer.step(lr, model, block_grads);
                }

                // Process input projection and output head
                let other_param_ids = model.get_other_param_ids();
                let other_grads =
                    GradientsParams::from_params(&mut grads, &model, &other_param_ids);
                model = optimizer.step(*other_lr, model, other_grads);

                model
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_xlstm_forward() {
        let device = Default::default();
        let config = XLstmconfig::new(64, 128, 2, 4, 32)
            .with_num_heads(4)
            .with_dropout(0.1)
            .with_lstm_type(LstmType::Alternate);

        let model = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = model.forward(input, None);

        assert_eq!(output.dims(), [4, 10, 32]);
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_xlstm_predict_last() {
        let device = Default::default();
        let config = XLstmconfig::new(64, 128, 2, 4, 1)
            .with_dropout(0.1)
            .with_lstm_type(LstmType::SLSTM);

        let model = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = model.predict_last(input, None);

        assert_eq!(output.dims(), [4, 1]);
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_xlstm_mixed_blocks() {
        let device = Default::default();
        let custom_types = alloc::vec![
            BlockType::SLSTM,
            BlockType::SLSTM,
            BlockType::MLSTM,
            BlockType::MLSTM,
        ];

        let config = XLstmconfig::new(64, 128, 2, 4, 32)
            .with_lstm_type(LstmType::Custom(custom_types.clone()));

        let model = config.init::<TestBackend>(&device);

        assert_eq!(model.get_block_config(), custom_types);
    }

    #[test]
    fn test_xlstm_no_projection() {
        let device = Default::default();
        let config = XLstmconfig::new(128, 128, 2, 2, 32).with_use_projection(false);

        let model = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 128], Distribution::Default, &device);

        let (output, _) = model.forward(input, None);

        assert_eq!(output.dims(), [4, 10, 32]);
    }
}