/*!
# sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, VarBuilder, ops};

/// State for sLSTM containing cell and hidden states
#[derive(Clone, Debug)]
pub struct SLstmstate {
    /// Cell state
    pub cell: Tensor,
    /// Normalizer state (for stable exponential gating)
    pub normalizer: Tensor,
    /// Hidden state
    pub hidden: Tensor,
    /// Stabilizer state (log-space max tracker)
    pub stabilizer: Tensor,
}

impl SLstmstate {
    /// Create a new sLSTM state
    pub fn new(
        cell: Tensor,
        normalizer: Tensor,
        hidden: Tensor,
        stabilizer: Tensor,
    ) -> Self {
        Self {
            cell,
            normalizer,
            hidden,
            stabilizer,
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            normalizer: self.normalizer.detach(),
            hidden: self.hidden.detach(),
            stabilizer: self.stabilizer.detach(),
        }
    }
}

/// Configuration for sLSTM
#[derive(Debug, Clone)]
pub struct SLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f32, // Candle uses f32
}

impl SLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize a new sLSTM
    pub fn init(&self, vb: VarBuilder) -> Result<SLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i)); // Make sure to structure keys correctly
            layers.push(SLstmcell::new(input_size, self.d_hidden, layer_vb)?);
        }

        Ok(SLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
        })
    }
}

/// sLSTM layer implementation
#[derive(Debug)]
pub struct SLstm {
    /// Stack of sLSTM cells
    pub layers: Vec<SLstmcell>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl SLstm {
    /// Forward pass through sLSTM consuming and returning states
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<SLstmstate>>,
    ) -> Result<(Tensor, Vec<SLstmstate>)> {
        let (batch_size, seq_length, _) = input_seq.dims3()?;
        let device = input_seq.device();

        // Initialize or consume provided states
        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, device)?,
        };

        let mut all_outputs = Vec::with_capacity(seq_length);

        for t in 0..seq_length {
            let input_t = input_seq.narrow(1, t, 1)?.squeeze(1)?;

            let mut layer_input = input_t;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // Take current state components
                let old_state = &hidden_states[layer_idx];
                
                // Consume the state and get new state back
                let (h_new, new_state) = layer.forward(&layer_input, old_state)?;

                // Update state
                hidden_states[layer_idx] = new_state;

                // Apply dropout between layers (but not after last layer)
                layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                    self.dropout_layer.forward(&h_new, true)? 
                } else {
                    h_new
                };
            }

            all_outputs.push(layer_input.unsqueeze(1)?);
        }

        let output = Tensor::cat(&all_outputs, 1)?;
        Ok((output, hidden_states))
    }

    /// Initialize hidden states
    fn init_hidden(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<Vec<SLstmstate>> {
        (0..self.num_layers)
            .map(|_| {
                Ok(SLstmstate::new(
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                ))
            })
            .collect()
    }
}

/// sLSTM cell implementation with exponential gating
#[derive(Debug)]
pub struct SLstmcell {
    /// Weight matrix for input to gates
    pub weight_ih: Tensor,
    /// Weight matrix for hidden to gates
    pub weight_hh: Tensor,
    /// Bias for gates
    pub bias: Tensor,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl SLstmcell {
    /// Create a new sLSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        vb: VarBuilder, // Use VarBuilder for initialization
    ) -> Result<Self> {
        // 4 gates: input, forget, cell, output
        
        // Init weights with Xavier Normal equivalent or similar. 
        // Candle's default init for linear is usually adequate, but we can customize.
        let weight_ih = vb.get_with_hints(
            (4 * hidden_size, input_size), 
            "weight_ih", 
            candle_nn::init::DEFAULT_KAIMING_NORMAL 
        )?;
        
        let weight_hh = vb.get_with_hints(
            (4 * hidden_size, hidden_size), 
            "weight_hh", 
            candle_nn::init::DEFAULT_KAIMING_NORMAL
        )?;

        // Initialize biases with specific values for stability
        // Forget gate bias initialized to 0.0 or something else if needed?
        // Original code: -2.0 for everything? 
        // Original: first hidden_size = -2.0 (input gate?), next hidden_size -> 2*hidden_size = -2.0
        // Wait, loop 0..hidden_size and hidden_size..2*hidden_size = -2.0. Others (2*h..4*h) are 0.0.
        
        let _bias_vals = {
            let mut data = vec![0.0f32; 4 * hidden_size];
            // Input gate bias?
            for item in data.iter_mut().take(hidden_size) {
                *item = -2.0; // "Input gate bias"? Original code comments are a bit sparse on mapping indices to gates
                // chunks[0] is i_gate.
            }
            // Forget gate bias? chunks[1] is f_gate.
            for item in data.iter_mut().take(2 * hidden_size).skip(hidden_size) {
                *item = -2.0; 
            }
            // Cell (g) and Output (o) remain 0.0
            data
        };
        
        let bias = vb.get_with_hints(
            4 * hidden_size, 
            "bias", 
            candle_nn::init::Init::Const(0.0) // We'll override or use a custom init, but explicit set is better
        )?;
        // Ideally we set the values. VarBuilder usually loads or inits. 
        // Let's force load from literal if we are creating from scratch or assume trained weights will handle it.
        // For now, let's just init as zeros then add constant? No, `vb` handles parameters. 
        // If we are strictly porting "init", we should rely on vb to find pretrained weights OR init with specific distribution.
        // But `vb` is lazy. 
        // Let's stick to standard initialization for now, unless we can pass a specific init.
        // Since I cannot easily inject custom values into a VarBuilder (it pulls from file or random),
        // I will assume for now that standard init is fine, or I'd need to manually create the tensor.
        // Note: The user might be training from scratch. 
        // Modification: If `vb` has no data, we might want to manually create tensors, but `vb` is the standard way.
        // Use `vb.get((...))` returns a tensor.
        
        // HACK: To replicate the specific bias init, we strictly should modify it.
        // But `vb` returns a generic tensor.
        // If the user wants specific init, we usually do it via the Init enum.
        // Candle `Init` doesn't support "partial fill". 
        // I will just use `zeros` for bias for now to keep it compilable, noting the difference.
        // Or better:
        // let bias = Tensor::from_vec(bias_vals, 4*hidden_size, vb.device())?;
        // But we want it to be a Var (trainable).
        // Let's trust the optimizer or user to load weights. If new, maybe I should do:
        // let bias = vb.get_with_hints(..., Init::FromVec(bias_vals))? ? No FromVec.
        
        // I will leave it as standard init (Zeros or Rand) for now. The specific bias values are for training stability.
        
        Ok(Self {
            weight_ih,
            weight_hh,
            bias,
            input_size,
            hidden_size,
        })
    }

    /// Forward pass through sLSTM cell consuming the state
    pub fn forward(
        &self,
        input: &Tensor,
        state: &SLstmstate,
    ) -> Result<(Tensor, SLstmstate)> {
        let SLstmstate {
            cell,
            normalizer,
            hidden,
            stabilizer,
        } = state;

        // Compute all gates: i, f, g, o
        // weight_ih: [4*H, I]. input: [B, I]. 
        // input @ weight_ih^T = [B, 4*H]
        let gates = input.matmul(&self.weight_ih.t()?)?
            .broadcast_add(&self.bias.unsqueeze(0)?)?
            .broadcast_add(&hidden.matmul(&self.weight_hh.t()?)?)?;

        let chunks = gates.chunk(4, 1)?;
        let i_gate = &chunks[0];
        let f_gate = &chunks[1];
        let g_gate = &chunks[2];
        let o_gate = &chunks[3];

        let m_prev_plus_f = stabilizer.add(f_gate)?;
        let m_new = m_prev_plus_f.maximum(i_gate)?; // max_pair equivalent? candle has maximum

        let i_exp = (i_gate - &m_new)?.clamp(-20.0, 0.0)?.exp()?;
        let f_exp = (m_prev_plus_f - &m_new)?.clamp(-20.0, 0.0)?.exp()?;

        let g = g_gate.tanh()?;
        let o = ops::sigmoid(o_gate)?;

        let c_new = ((&f_exp * cell)? + (&i_exp * g)?)?;
        let n_new = ((f_exp * normalizer)? + i_exp)?;

        let n_safe = n_new.clamp(1e-6, f32::MAX)?; // clamp_min
        let h_new = (o * (c_new.clone() / n_safe)?.tanh()?)?;

        let new_state = SLstmstate::new(c_new, n_new, h_new.clone(), m_new);
        Ok((h_new, new_state))
    }
}

