/*!
# sLSTM: Scalar Long Short-Term Memory
Implementation according to: "xLSTM: Extended Long Short-Term Memory" (2405.04517v2)

The sLSTM (Scalar LSTM) extends the traditional LSTM with:
1. Exponential gating with a log-space max-stabilizer (m_t).
2. Normalization (n_t) of the cell state to maintain boundedness.
3. Memory mixing via the hidden-to-state recurrent connection.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, VarBuilder, ops};

/// State for sLSTM containing cell, normalizer, hidden, and stabilizer states
#[derive(Clone, Debug)]
pub struct SLstmstate {
    /// Cell state (c_t)
    pub cell: Tensor,
    /// Normalizer state (n_t)
    pub normalizer: Tensor,
    /// Hidden state (h_t)
    pub hidden: Tensor,
    /// Stabilizer state (m_t: tracking the maximum gate logit)
    pub stabilizer: Tensor,
}

impl SLstmstate {
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
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
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

    pub fn init(&self, vb: VarBuilder) -> Result<SLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i));
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

/// sLSTM layer stack implementation
#[derive(Debug)]
pub struct SLstm {
    pub layers: Vec<SLstmcell>,
    pub dropout_layer: Dropout,
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
}

impl SLstm {
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<SLstmstate>>,
    ) -> Result<(Tensor, Vec<SLstmstate>)> {
        let (batch_size, seq_length, _) = input_seq.dims3()?;
        let device = input_seq.device();

        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, device)?,
        };

        let mut all_outputs = Vec::with_capacity(seq_length);

        for t in 0..seq_length {
            let input_t = input_seq.narrow(1, t, 1)?.squeeze(1)?;
            let mut layer_input = input_t;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (h_new, new_state) = layer.forward(&layer_input, &hidden_states[layer_idx])?;
                hidden_states[layer_idx] = new_state;

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

    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<SLstmstate>> {
        (0..self.num_layers)
            .map(|_| {
                Ok(SLstmstate::new(
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    // xLSTM paper: n_0 = 0 (Equation 13 & 14 starting sum from 0)
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                ))
            })
            .collect()
    }
}

/// sLSTM cell implementation with Exponential Gating (Eq. 15-20)
#[derive(Debug)]
pub struct SLstmcell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl SLstmcell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        vb: VarBuilder, 
    ) -> Result<Self> {
        // xLSTM paper suggests standard initialization for LSTM-like gates.
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

        // Bias 0.0 means e^0 = 1.0 initially for i and f gates.
        let bias = vb.get_with_hints(
            4 * hidden_size, 
            "bias", 
            candle_nn::init::Init::Const(0.0)
        )?;

        Ok(Self {
            weight_ih,
            weight_hh,
            bias,
            input_size,
            hidden_size,
        })
    }

    /// Forward pass through sLSTM cell (Single Step)
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

        // 1. Projections for i, f, z, o gates [B, 4*H]
        let gates = input.matmul(&self.weight_ih.t()?)?
            .broadcast_add(&self.bias.unsqueeze(0)?)?
            .broadcast_add(&hidden.matmul(&self.weight_hh.t()?)?)?;

        // 2. Gate separation (Assuming order: i, f, z, o)
        let chunks = gates.chunk(4, 1)?;
        let i_gate = &chunks[0]; // Logit del Input gate
        let f_gate = &chunks[1]; // Logit del Forget gate
        let z_gate = &chunks[2]; // Logit del Input content (phi)
        let o_gate = &chunks[3]; // Logit del Output gate (sigma)

        // 3. Log-Space Max Stabilization (Eq. 18-19)
        // m_t = max(f_gate + m_prev, i_gate)
        let m_prev_plus_f = stabilizer.add(f_gate)?;
        let m_new = m_prev_plus_f.maximum(i_gate)?; 

        // 4. Stabilized Exponential Gating
        // i_t' = exp(i_gate - m_t)
        // f_t' = exp(f_gate + m_prev - m_t)
        let i_exp = (i_gate - &m_new)?.clamp(-30.0, 0.0)?.exp()?;
        let f_exp = (m_prev_plus_f - &m_new)?.clamp(-30.0, 0.0)?.exp()?;

        // Activaciones no-exponenciales
        let z = z_gate.tanh()?;      // Input content (z en el paper)
        let o = ops::sigmoid(o_gate)?; // Output gate (sigma)

        // 5. State Updates (Eq. 13-14)
        // c_t = f_t' * c_prev + i_t' * z_t
        // n_t = f_t' * n_prev + i_t'
        let c_new = ((&f_exp * cell)? + (&i_exp * z)?)?;
        let n_new = ((f_exp * normalizer)? + i_exp)?;

        // 6. Normalization and Output (Eq. 16-17)
        // Hidden state h_t = o_t * (c_t / n_t)
        // xLSTM paper: "The normalization ensures that h_t is bounded, making tanh unnecessary."
        let n_safe = n_new.clamp(1e-6, f32::MAX)?; 
        let h_new = (o * (c_new.clone() / n_safe)?)?;

        let new_state = SLstmstate::new(c_new, n_new, h_new.clone(), m_new);
        Ok((h_new, new_state))
    }
}
