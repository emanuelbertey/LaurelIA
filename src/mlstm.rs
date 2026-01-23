/*!
# mLSTM: Matrix Long Short-Term Memory
Implementation according to: "xLSTM: Extended Long Short-Term Memory" (2405.04517v2)
Section 2.3 and Appendix A.3 (Parallel Dual Form).

The mLSTM replaces the scalar memory of a standard LSTM with a matrix memory (C_t),
offering higher storage capacity. It uses:
1. Multi-Head Structure (similar to Transformers).
2. Covariance-like update for the memory matrix.
3. Stabilized exponential gating for all-parallel computation.
*/

use candle_core::{Tensor, Result};
use candle_nn::{Dropout, Module, VarBuilder, Linear, ops, linear};

/// State for mLSTM (Matrix Memory)
#[derive(Clone, Debug)]
pub struct MLstmstate {
    /// Matrix memory cell shape: [B, H, D_h, D_h]
    pub cell: Tensor,       
    /// Normalizer state shape: [B, H, D_h]
    pub normalizer: Tensor, 
    /// Last max logit for stability shape: [B, H, 1]
    pub m_t: Tensor,        
}

impl MLstmstate {
    pub fn new(cell: Tensor, normalizer: Tensor, m_t: Tensor) -> Self {
        Self { cell, normalizer, m_t }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            normalizer: self.normalizer.detach(),
            m_t: self.m_t.detach(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub expansion_factor: usize,
    pub dropout: f32,
}

impl MLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            num_heads,
            expansion_factor: 2,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_expansion_factor(mut self, factor: usize) -> Self {
        self.expansion_factor = factor;
        self
    }

    pub fn init(&self, vb: VarBuilder) -> Result<MLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let layer_vb = vb.pp(format!("layer_{}", i));
            layers.push(MLstmcell::new(self.d_input, self.d_hidden, self.num_heads, self.expansion_factor, layer_vb)?);
        }

        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            num_layers: self.num_layers,
        })
    }
}

#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout_layer: Dropout,
    pub num_layers: usize,
}

impl MLstm {
    pub fn forward(&self, input: &Tensor, states: Option<Vec<MLstmstate>>) -> Result<(Tensor, Vec<MLstmstate>)> {
        let mut current_input = input.clone();
        let mut new_states = Vec::with_capacity(self.num_layers);
        
        for (i, layer) in self.layers.iter().enumerate() {
            let state_ref = states.as_ref().map(|s| &s[i]);
            let (output, next_state) = layer.forward_sequence(&current_input, state_ref)?;
            current_input = self.dropout_layer.forward(&output, true)?;
            new_states.push(next_state);
        }

        Ok((current_input, new_states))
    }
}

#[derive(Debug)]
pub struct MLstmcell {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_i: Linear,
    pub w_f: Linear,
    pub w_o: Linear,
    pub w_down: Linear,
    
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_inner: usize,
}

impl MLstmcell {
    pub fn new(d_in: usize, d_hid: usize, n_heads: usize, exp_factor: usize, vb: VarBuilder) -> Result<Self> {
        let d_inner = d_hid * exp_factor;
        let head_dim = d_inner / n_heads;

        Ok(Self {
            w_q: linear(d_in, d_inner, vb.pp("w_q"))?,
            w_k: linear(d_in, d_inner, vb.pp("w_k"))?,
            w_v: linear(d_in, d_inner, vb.pp("w_v"))?,
            w_i: linear(d_in, d_inner, vb.pp("w_i"))?,
            w_f: linear(d_in, d_inner, vb.pp("w_f"))?,
            w_o: linear(d_in, d_inner, vb.pp("w_o"))?,
            w_down: linear(d_inner, d_hid, vb.pp("w_down"))?,
            num_heads: n_heads,
            head_dim,
            d_inner,
        })
    }

    /// Parallel Forward Pass (Equations 79-86)
    pub fn forward_sequence(&self, x: &Tensor, state_prev: Option<&MLstmstate>) -> Result<(Tensor, MLstmstate)> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let dev = x.device();

        // 1. Projections (Q, K, V)
        // Shapes: [B, H, S, D_h]
        let q = self.w_q.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = self.w_k.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = self.w_v.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        
        // Log-gates (Eq. 25-26)
        let i_tilde = self.w_i.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?;
        let f_tilde = self.w_f.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?;
        let o_gate = ops::sigmoid(&self.w_o.forward(x)?)?;

        // Gate scalars per head [B, H, S]
        // mean(3) reduces the D_head dimension, resulting in [B, H, S].
        // This preserves the head dimension (dim 1), ensuring each head has its own gate.
        let log_i = i_tilde.mean(3)?; 

        // Stable log-sigmoid for forget gate:
        // log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        // We use the raw f_tilde (averaged over D_head to get scalar per head)
        let f_gate_scalar = f_tilde.mean(3)?;
        
      /*  let log_f = f_gate_scalar.neg()?.broadcast_as(f_gate_scalar.shape())?
            .exp()?.log_1p()?.neg()?;
*/
          let log_f = f_gate_scalar
            .neg()?           // -x
            .exp()?           // exp(-x)
            .affine(1.0, 1.0)? // 1 + exp(-x)
            .log()?           // log(1 + exp(-x))
            .neg()?;          // -log(1 + exp(-x))  
        
        // 2. Parallel Exponential Gating (Dual Form)
        // Forget gate cumulative sum: s_i = sum_{j=1}^i log_f_j
        let s = log_f.cumsum(2)?; 
        
        // log_D [B, H, S, S]: log_D[i, j] = log_i[j] + s[i] - s[j]
        let log_d = s.unsqueeze(3)? 
            .broadcast_sub(&s.unsqueeze(2)?)? 
            .broadcast_add(&log_i.unsqueeze(2)?)?; 

        // Causal Masking
        let indices = Tensor::arange(0u32, seq_len as u32, dev)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?
            .ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?
            .unsqueeze(0)?.unsqueeze(0)?;
            
        let neg_inf = Tensor::new(-1e10f32, dev)?.broadcast_as(log_d.shape())?;
        let log_d_masked = mask.broadcast_as(log_d.shape())?.where_cond(&log_d, &neg_inf)?;

        // Stabilizer m_t (Eq. 80-81) 
        let m_current = log_d_masked.max_keepdim(3)?; // [B, H, S, 1]
        
        // m_initial = s_i + m_prev
        let m = if let Some(state) = state_prev {
            let m_prev = state.m_t.unsqueeze(2)?.broadcast_as((b_sz, self.num_heads, 1, 1))?; 
            let m_initial = s.unsqueeze(3)?.broadcast_add(&m_prev)?;
            m_initial.maximum(&m_current)?
        } else {
            m_current
        };
        
        // Stabilized weights d_prime = exp(log_D - m)
        let d_prime = log_d_masked.broadcast_sub(&m)?.exp()?; 

        // 3. Retrieval (Eq. 82-86)
        let scale = Tensor::new((self.head_dim as f32).sqrt(), dev)?;
        let q_scaled = q.broadcast_div(&scale)?;
        
        // h_parallel = (D' @ V) ? No, we need the matrix form property:
        // Proper parallel hidden state retrieval
        let qk_t = q_scaled.matmul(&k.transpose(2, 3)?)?;
        let matrix_weights = qk_t.broadcast_mul(&d_prime)?;
        let h_raw = matrix_weights.matmul(&v)?; 

        // n_parallel = D' @ K
        let n_vector_seq = matrix_weights.matmul(&k)?; // Vector normalizer [B, H, S, D_h]

        // Handle initial state contribution
        let (total_h_raw, total_n_vector) = if let Some(state) = state_prev {
            // f_init = exp(s_i + m_prev - m_i)
            let m_prev = state.m_t.unsqueeze(2)?.broadcast_as((b_sz, self.num_heads, 1, 1))?;
            let f_initial = s.unsqueeze(3)?.broadcast_add(&m_prev)?.broadcast_sub(&m)?.exp()?;
            
            let h_initial = q_scaled.matmul(&state.cell)?.broadcast_mul(&f_initial)?;
            let n_initial = state.normalizer.unsqueeze(2)?.broadcast_mul(&f_initial)?;
            
            ((h_raw + h_initial)?, (n_vector_seq + n_initial)?)
        } else {
            (h_raw, n_vector_seq)
        };

        // Retrieval Denominator: n_i^T @ q_i (Eq. 21)
        let n_transpose_q = total_n_vector.broadcast_mul(&q_scaled)?.sum_keepdim(3)?;
        let n_safe = n_transpose_q.abs()?.maximum(&(m.neg()?.exp()?))?;
        
        let h_norm = total_h_raw.broadcast_div(&n_safe)?;

        // 4. State Update for continuity (Last step)
        let last_idx = seq_len - 1;
        let last_m = m.narrow(2, last_idx, 1)?.squeeze(3)?;
        let final_norm = total_n_vector.narrow(2, last_idx, 1)?.squeeze(2)?;

        let final_cell = if let Some(state) = state_prev {
             let m_prev = state.m_t.unsqueeze(2)?.broadcast_as((b_sz, self.num_heads, 1, 1))?;
             let f_last = s.narrow(2, last_idx, 1)?.unsqueeze(3)?.broadcast_add(&m_prev)?.broadcast_sub(&m.narrow(2, last_idx, 1)?)?.exp()?;
             
             let last_row_weights = d_prime.narrow(2, last_idx, 1)?; // [B, H, 1, S]
             let weighted_v = v.broadcast_mul(&last_row_weights.transpose(2, 3)?)?; 
             let cell_update = weighted_v.transpose(2, 3)?.matmul(&k)?; 
             
             (state.cell.broadcast_mul(&f_last)? + cell_update)?
        } else {
             let last_row_weights = d_prime.narrow(2, last_idx, 1)?;
             let weighted_v = v.broadcast_mul(&last_row_weights.transpose(2, 3)?)?;
             weighted_v.transpose(2, 3)?.matmul(&k)?
        };

        // 5. Output Projection
        let h_combined = h_norm.permute((0, 2, 1, 3))?.reshape((b_sz, seq_len, ()))?;
        let out = self.w_down.forward(&(o_gate * h_combined)?)?;

        let final_state = MLstmstate::new(final_cell, final_norm, last_m);

        Ok((out, final_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn test_mlstm_shapes() -> Result<()> {
        let device = Device::Cpu;
        let b_sz = 2;
        let seq_len = 8;
        let d_in = 16;
        let d_hid = 32;
        let n_heads = 4;
        
        let config = MLstmconfig::new(d_in, d_hid, 1, n_heads);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = config.init(vb)?;
        
        let input = Tensor::randn(0f32, 1f32, (b_sz, seq_len, d_in), &device)?;
        let (output, states) = model.forward(&input, None)?;
        
        assert_eq!(output.dims(), &[b_sz, seq_len, d_hid]);
        assert_eq!(states.len(), 1);
        
        // head_dim = (d_hid * expansion_factor) / n_heads = (32 * 2) / 4 = 16
        let head_dim = 16;
        assert_eq!(states[0].cell.dims(), &[b_sz, n_heads, head_dim, head_dim]);
        assert_eq!(states[0].normalizer.dims(), &[b_sz, n_heads, head_dim]);
        assert_eq!(states[0].m_t.dims(), &[b_sz, n_heads, 1]);
        
        Ok(())
    }

    #[test]
    fn test_mlstm_continuity() -> Result<()> {
        let device = Device::Cpu;
        let b_sz = 1;
        let d_in = 8;
        let d_hid = 16;
        let n_heads = 2;
        
        let config = MLstmconfig::new(d_in, d_hid, 1, n_heads);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = config.init(vb)?;
        
        // First half of sequence
        let input1 = Tensor::randn(0f32, 1f32, (b_sz, 4, d_in), &device)?;
        let (_, states1) = model.forward(&input1, None)?;
        
        // Second half of sequence
        let input2 = Tensor::randn(0f32, 1f32, (b_sz, 4, d_in), &device)?;
        let (output2, _states2) = model.forward(&input2, Some(states1.clone()))?;
        
        // Verify output is different when state is injected
        let (output2_no_state, _) = model.forward(&input2, None)?;
        
        let diff = (output2 - output2_no_state)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff > 1e-5, "State injection should change the output: diff = {}", diff);
        
        Ok(())
    }
}
