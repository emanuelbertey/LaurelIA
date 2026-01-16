/*!
# xLSTM: Extended Long Short-Term Memory Model

This module implements the main xLSTM model that can stack multiple blocks
with flexible mixing of sLSTM and mLSTM, including support for per-block
learning rates.
*/

use candle_core::{Tensor, Result};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, Embedding, layer_norm, linear, embedding};
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
    Custom(Vec<BlockType>),
}

/// Configuration for xLSTM model
#[derive(Debug, Clone)]
pub struct XLstmconfig {
    /// Input size (number of features or embedding dimension)
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
    pub dropout: f32,
    /// Whether to use bidirectional LSTM
    pub bidirectional: bool,
    /// Number of heads for mLSTM blocks
    pub num_heads: usize,
    /// LSTM type configuration
    pub lstm_type: LstmType,
    /// Whether to use input projection
    pub use_projection: bool,
    /// Vocabulary size (if using embeddings)
    pub vocab_size: Option<usize>,
}

impl XLstmconfig {
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        num_layers: usize, 
        num_blocks: usize, 
        output_size: usize
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers,
            num_blocks,
            output_size,
            dropout: 0.0,
            bidirectional: false,
            num_heads: 4,
            lstm_type: LstmType::SLSTM,
            use_projection: true,
            vocab_size: None,
        }
    }

    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
    
    pub fn with_lstm_type(mut self, lstm_type: LstmType) -> Self {
        self.lstm_type = lstm_type;
        self
    }
    
    pub fn with_use_projection(mut self, use_projection: bool) -> Self {
        self.use_projection = use_projection;
        self
    }

    /// Initialize a new xLSTM model
    pub fn init(&self, vb: VarBuilder) -> Result<XLstm> {
        // Parse block types
        let block_types = self.parse_lstm_type();

        // Create embedding if requested
        let embedding = if let Some(vocab) = self.vocab_size {
            Some(embedding(vocab, self.input_size, vb.pp("embedding"))?)
        } else {
            None
        };

        // Create input projection if requested
        let (input_projection, block_input_size) = if self.use_projection {
            let linear = linear(self.input_size, self.hidden_size, vb.pp("proj_linear"))?;
            let norm = layer_norm(self.hidden_size, 1e-5, vb.pp("proj_norm"))?;
            let dropout = Dropout::new(self.dropout);
            (Some((linear, norm, dropout)), self.hidden_size)
        } else {
            (None, self.input_size)
        };

        // Create blocks
        let mut blocks = Vec::with_capacity(self.num_blocks);
        for (i, &block_type) in block_types.iter().enumerate() {
            let config = XLstmblockConfig::new(
                    block_input_size,
                    self.hidden_size,
                    self.num_layers,
                    block_type,
                )
                .with_num_heads(self.num_heads)
                .with_dropout(self.dropout);
            // .with_initializer(self.initializer.clone()) // Handled by standard init
            
            blocks.push(config.init(vb.pp(format!("block_{}", i)))?);
        }

        // Create output head
        let head_linear1 = linear(block_input_size, self.hidden_size, vb.pp("head_linear1"))?;
        let head_dropout = Dropout::new(self.dropout);
        let head_linear2 = linear(self.hidden_size, self.output_size, vb.pp("head_linear2"))?;
        
        let output_head = (
            head_linear1,
            head_dropout,
            head_linear2,
        );

        Ok(XLstm {
            embedding,
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
        })
    }

    fn parse_lstm_type(&self) -> Vec<BlockType> {
        match &self.lstm_type {
            LstmType::SLSTM => vec![BlockType::SLSTM; self.num_blocks],
            LstmType::MLSTM => vec![BlockType::MLSTM; self.num_blocks],
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
#[derive(Debug)]
pub struct XLstm {
    /// Optional embedding layer
    pub embedding: Option<Embedding>,
    /// Optional input projection layers
    pub input_projection: Option<(Linear, LayerNorm, Dropout)>,
    /// Stack of xLSTM blocks
    pub blocks: Vec<XLstmblock>,
    /// Output head layers
    pub output_head: (Linear, Dropout, Linear),
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
    pub dropout: f32,
    /// Whether input projection is used
    pub use_projection: bool,
}

impl XLstm {
    /// Forward pass through xLSTM model
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<Option<LSTMState>>>,
    ) -> Result<(Tensor, Vec<Option<LSTMState>>)> {
        // Apply embedding if present
        let mut x = if let Some(embedding) = &self.embedding {
            embedding.forward(input_seq)?
        } else {
            input_seq.clone()
        };

        // Apply input projection if present
        if let Some((linear, norm, dropout)) = &self.input_projection {
            x = linear.forward(&x)?;
            x = norm.forward(&x)?;
            x = x.gelu()?;
            x = dropout.forward(&x, true)?;
        }

        // Initialize states if not provided
        let mut hidden_states = states.unwrap_or_else(|| vec![None; self.num_blocks]);

        // Pass through blocks
        for (i, block) in self.blocks.iter().enumerate() {
            let old_state = hidden_states[i].take();
            let (output, new_state) = block.forward(&x, old_state)?;
            x = output;
            hidden_states[i] = new_state;
        }

        // Apply output head
        let (linear1, dropout, linear2) = &self.output_head;
        x = linear1.forward(&x)?;
        x = x.gelu()?;
        x = dropout.forward(&x, true)?;
        let output = linear2.forward(&x)?;

        Ok((output, hidden_states))
    }

    /// Forward pass returning only the last timestep prediction
    pub fn predict_last(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<Option<LSTMState>>>,
    ) -> Result<(Tensor, Vec<Option<LSTMState>>)> {
        let (output, states) = self.forward(input_seq, states)?;
        let (_batch_size, seq_length, _) = output.dims3()?;
        let last_output = output
            .narrow(1, seq_length - 1, 1)?
            .squeeze(1)?;
        Ok((last_output, states))
    }

    /// Get block configuration
    pub fn get_block_config(&self) -> Vec<BlockType> {
        self.blocks
            .iter()
            .map(XLstmblock::get_type)
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
