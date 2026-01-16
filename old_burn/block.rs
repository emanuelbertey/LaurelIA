/*!
# xLSTM Block Implementation

This module implements the xLSTM block as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM block combines either sLSTM or mLSTM with layer normalization,
residual connections, and additional linear projections.
*/

use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{MLstm, MLstmconfig, MLstmstate, SLstm, SLstmconfig, SLstmstate};

/// Type of LSTM block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    /// Scalar LSTM
    SLSTM,
    /// Matrix LSTM
    MLSTM,
}

/// Configuration for xLSTM block
#[derive(Config, Debug)]
pub struct XLstmblockConfig {
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    #[config(default = "4")]
    pub num_heads: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Block type (sLSTM or mLSTM)
    pub block_type: BlockType,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:0.1}")]
    pub initializer: Initializer,
}

impl XLstmblockConfig {
    /// Initialize a new xLSTM block
    pub fn init<B: Backend>(&self, device: &B::Device) -> XLstmblock<B> {
        match self.block_type {
            BlockType::SLSTM => {
                let lstm: SLstm<B> =
                    SLstmconfig::new(self.input_size, self.hidden_size, self.num_layers)
                        .with_dropout(self.dropout)
                        .with_initializer(self.initializer.clone())
                        .init(device);

                XLstmblock {
                    lstm: LSTMVariant::SLSTM(lstm),
                    norm: LayerNormConfig::new(self.hidden_size).init(device),
                    dropout: DropoutConfig::new(self.dropout).init(),
                    proj: LinearConfig::new(self.hidden_size, self.input_size).init(device),
                }
            }
            BlockType::MLSTM => {
                let lstm: MLstm<B> =
                    MLstmconfig::new(self.input_size, self.hidden_size, self.num_layers)
                        .with_num_heads(self.num_heads)
                        .with_dropout(self.dropout)
                        .with_initializer(self.initializer.clone())
                        .init(device);

                XLstmblock {
                    lstm: LSTMVariant::MLSTM(lstm),
                    norm: LayerNormConfig::new(self.hidden_size).init(device),
                    dropout: DropoutConfig::new(self.dropout).init(),
                    proj: LinearConfig::new(self.hidden_size, self.input_size).init(device),
                }
            }
        }
    }
}

/// Enum to hold either sLSTM or mLSTM
#[derive(Module, Debug)]
pub enum LSTMVariant<B: Backend> {
    /// Scalar LSTM variant
    SLSTM(SLstm<B>),
    /// Matrix LSTM variant
    MLSTM(MLstm<B>),
}

/// Enum for holding either sLSTM or mLSTM states
#[derive(Debug, Clone)]
pub enum LSTMState<B: Backend> {
    /// States for sLSTM
    SLSTM(alloc::vec::Vec<SLstmstate<B, 2>>),
    /// States for mLSTM
    MLSTM(alloc::vec::Vec<MLstmstate<B>>),
}

/// xLSTM block combining LSTM with normalization and projections
#[derive(Module, Debug)]
pub struct XLstmblock<B: Backend> {
    /// LSTM variant (sLSTM or mLSTM)
    pub lstm: LSTMVariant<B>,
    /// Layer normalization
    pub norm: LayerNorm<B>,
    /// Dropout layer
    pub dropout: Dropout,
    /// Projection layer
    pub proj: Linear<B>,
}

impl<B: Backend> XLstmblock<B> {
    /// Forward pass through xLSTM block
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor [`batch_size`, `seq_length`, `input_size`]
    /// * `state` - Optional initial state
    ///
    /// # Returns
    /// * Output tensor [`batch_size`, `seq_length`, `input_size`]
    /// * Final state
    pub fn forward(
        &self,
        input_seq: Tensor<B, 3>,
        state: Option<LSTMState<B>>,
    ) -> (Tensor<B, 3>, Option<LSTMState<B>>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive,
        B: Backend<FloatElem: num_traits::FromPrimitive>,
    {
        let (lstm_output, new_state): (Tensor<B, 3>, Option<LSTMState<B>>) =
            match (&self.lstm, state) {
                (LSTMVariant::SLSTM(lstm), Some(LSTMState::SLSTM(s))) => {
                    let (out, state): (Tensor<B, 3>, Vec<SLstmstate<B, 2>>) =
                        lstm.forward(&input_seq, Some(s)); // No clone here
                    (out, Some(LSTMState::SLSTM(state)))
                }
                (LSTMVariant::SLSTM(lstm), _) => {
                    let (out, state) = lstm.forward(&input_seq, None);
                    (out, Some(LSTMState::SLSTM(state)))
                }
                (LSTMVariant::MLSTM(lstm), Some(LSTMState::MLSTM(s))) => {
                    let (out, state) = lstm.forward(&input_seq, Some(s));
                    (out, Some(LSTMState::MLSTM(state)))
                }
                (LSTMVariant::MLSTM(lstm), _) => {
                    let (out, state) = lstm.forward(&input_seq, None);
                    (out, Some(LSTMState::MLSTM(state)))
                }
            };

        // Apply normalization BEFORE activation for stability (critical for mLSTM)
        let output: Tensor<B, 3> = self.norm.forward(lstm_output);
        // Apply activation
        let output: Tensor<B, 3> = activation::gelu(output);
        // Apply projection
        let output: Tensor<B, 3> = self.proj.forward(output);
        // Apply dropout
        let output: Tensor<B, 3> = self.dropout.forward(output);
        // Residual connection
        let output: Tensor<B, 3> = output + input_seq;

        (output, new_state)
    }

    /// Get the block type
    pub const fn get_type(&self) -> BlockType {
        match &self.lstm {
            LSTMVariant::SLSTM(_) => BlockType::SLSTM,
            LSTMVariant::MLSTM(_) => BlockType::MLSTM,
        }
    }
}



