/*!
# xLSTM Block Implementation

This module implements the xLSTM block as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM block combines either sLSTM or mLSTM with layer normalization,
residual connections, and additional linear projections.
*/

use candle_core::{Tensor, Result};
use candle_nn::{Dropout, Module, VarBuilder, LayerNorm, Linear, layer_norm, linear};
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
#[derive(Debug, Clone)]
pub struct XLstmblockConfig {
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Block type (sLSTM or mLSTM)
    pub block_type: BlockType,

}

impl XLstmblockConfig {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, block_type: BlockType) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers,
            num_heads: 4,
            dropout: 0.0,
            block_type,
        }
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize a new xLSTM block
    pub fn init(&self, vb: VarBuilder) -> Result<XLstmblock> {
        let norm = layer_norm(self.input_size, 1e-5, vb.pp("norm"))?;
        
        let lstm = match self.block_type {
            BlockType::SLSTM => {
                let lstm_config = SLstmconfig::new(self.input_size, self.hidden_size, self.num_layers)
                    .with_dropout(self.dropout);
                let lstm = lstm_config.init(vb.pp("lstm"))?;
                LSTMVariant::SLSTM(lstm)
            }
            BlockType::MLSTM => {
                let lstm_config = MLstmconfig::new(self.input_size, self.hidden_size, self.num_layers, self.num_heads)
                    .with_dropout(self.dropout);
                let lstm = lstm_config.init(vb.pp("lstm"))?;
                LSTMVariant::MLSTM(lstm)
            }
        };

        let proj = linear(self.hidden_size, self.input_size, vb.pp("proj"))?;
        let dropout = Dropout::new(self.dropout);

        Ok(XLstmblock {
            lstm,
            norm,
            dropout,
            dropout_prob: self.dropout,
            proj,
        })
    }
}

/// Enum to hold either sLSTM or mLSTM
#[derive(Debug)]
pub enum LSTMVariant {
    /// Scalar LSTM variant
    SLSTM(SLstm),
    /// Matrix LSTM variant
    MLSTM(MLstm),
}

/// Enum for holding either sLSTM or mLSTM states
#[derive(Debug, Clone)]
pub enum LSTMState {
    /// States for sLSTM
    SLSTM(Vec<SLstmstate>),
    /// States for mLSTM
    MLSTM(Vec<MLstmstate>),
}

impl LSTMState {
    pub fn detach(&self) -> Self {
        match self {
            LSTMState::SLSTM(v) => LSTMState::SLSTM(v.iter().map(|s| s.detach()).collect()),
            LSTMState::MLSTM(v) => LSTMState::MLSTM(v.iter().map(|m| m.detach()).collect()),
        }
    }
}

/// xLSTM block combining LSTM with normalization and projections
#[derive(Debug)]
pub struct XLstmblock {
    /// LSTM variant (sLSTM or mLSTM)
    pub lstm: LSTMVariant,
    /// Layer normalization
    pub norm: LayerNorm,
    /// Dropout layer
    pub dropout: Dropout,
    /// Projection layer
    pub proj: Linear,
        // dropout dinamico 
    pub dropout_prob: f32,
}

impl XLstmblock {
    pub fn forward(
        &self,
        input_seq: &Tensor,
        state: Option<LSTMState>,
    ) -> Result<(Tensor, Option<LSTMState>)> {
        
        // 1. PRE-NORM: Vital para apilar muchos bloques. 
        // Normalizamos los datos ANTES de que entren a la lógica pesada.
        let x = self.norm.forward(input_seq)?;

        // 2. DROPOUT INICIAL: Solo si es necesario sobre el input normalizado
        let x = if self.dropout_prob > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout_prob)?
        } else {
            x
        };

        // 3. PASO POR LA VARIANTE (mLSTM o sLSTM)
        let (lstm_output, new_state) = match (&self.lstm, state) {
            (LSTMVariant::SLSTM(lstm), s) => {
                let s_val = match s { Some(LSTMState::SLSTM(st)) => Some(st), _ => None };
                let (out, state) = lstm.forward(&x, s_val)?;
                (out, Some(LSTMState::SLSTM(state)))
            }
            (LSTMVariant::MLSTM(lstm), s) => {
                let s_val = match s { Some(LSTMState::MLSTM(st)) => Some(st), _ => None };
                let (out, state) = lstm.forward(&x, s_val)?;
                (out, Some(LSTMState::MLSTM(state)))
            }
            _ => candle_core::bail!("Mismatched state/variant"),
        };

        // 4. PROYECCIÓN Y DROPOUT DE SALIDA
        let output = self.proj.forward(&lstm_output)?;
        
        // 5. RESIDUAL CONNECTION (Pre-Norm Style)
        // Sumamos el resultado a 'input_seq' original (el que no fue normalizado)
        // Esto crea el "highway" de gradientes limpio.
       // let output = (output + input_seq)?;
       let output = ((output * 0.8)? + input_seq)?;

        Ok((output, new_state))
    }


    /// Get the block type
    pub fn get_type(&self) -> BlockType {
        match &self.lstm {
            LSTMVariant::SLSTM(_) => BlockType::SLSTM,
            LSTMVariant::MLSTM(_) => BlockType::MLSTM,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};
    use crate::{SLstmstate, MLstmstate};

    #[test]
    fn test_lstm_state_detach() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        // Test SLSTM detach
        let s_state = SLstmstate::new(
            Tensor::zeros((1, 10), DType::F32, &device)?,
            Tensor::zeros((1, 10), DType::F32, &device)?,
            Tensor::zeros((1, 10), DType::F32, &device)?,
            Tensor::zeros((1, 10), DType::F32, &device)?,
        );
        let lstm_state = LSTMState::SLSTM(vec![s_state]);
        let detached = lstm_state.detach();
        if let LSTMState::SLSTM(states) = detached {
             assert_eq!(states.len(), 1);
        } else {
             panic!("Wrong variant");
        }

        // Test MLSTM detach
        let m_state = MLstmstate::new(
             Tensor::zeros((1, 4, 16, 16), DType::F32, &device)?,
             Tensor::zeros((1, 10), DType::F32, &device)?,
             Tensor::zeros((1, 4, 16), DType::F32, &device)?,
             Tensor::zeros((1, 4, 1), DType::F32, &device)?,
        );
        let lstm_state_m = LSTMState::MLSTM(vec![m_state]);
        let detached_m = lstm_state_m.detach();
        if let LSTMState::MLSTM(states) = detached_m {
             assert_eq!(states.len(), 1);
        } else {
             panic!("Wrong variant");
        }
        
        Ok(())
    }
}