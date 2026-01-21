/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

This implementation strictly follows the parallel (Dual Form) architecture with:
1. Q, K, V projections.
2. Gates (f, i) projecting from concatenated [Q, K, V].
3. Matrix-valued state (C_t) and normalizer (n_t).
4. Exponential gating with max-stabilization (m_t).
           */

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, linear, layer_norm, init::Init};

/// State for mLSTM containing cell matrix, normalizer, and max stabilization values
#[derive(Clone, Debug)]
pub struct MLstmstate {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor,
    /// Hidden state - placeholder or last sequence output
    pub hidden: Tensor,
    /// Normalizer state - vector of shape [`batch_size`, `num_heads`, 1, `head_dim`]
    pub normalizer: Tensor,
    /// Max gate log state for numeric stability - shape [`batch_size`, `num_heads`, 1, 1]
    pub max_gate_log: Tensor,
}

impl MLstmstate {
    pub fn new(cell: Tensor, hidden: Tensor, normalizer: Tensor, max_gate_log: Tensor) -> Self {
        Self { cell, hidden, normalizer, max_gate_log }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
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

    pub fn with_expansion_factor(mut self, factor: usize) -> Self {
        self.expansion_factor = factor;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }


    pub fn init(&self, vb: VarBuilder) -> Result<MLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            layers.push(MLstmcell::new(input_size, self.d_hidden, self.num_heads, self.expansion_factor, vb.pp(format!("layer_{}", i)))?);
        }
        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
        })
    }
}

#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout_layer: Dropout,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
}

impl MLstm {
    pub fn forward(&self, input_seq: &Tensor, states: Option<Vec<MLstmstate>>) -> Result<(Tensor, Vec<MLstmstate>)> {
        let (batch_size, _, _) = input_seq.dims3()?;
        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, input_seq.device())?,
        };
        let mut x = input_seq.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, new_state) = layer.forward_sequence(&x, &hidden_states[i])?;
            hidden_states[i] = new_state;
            x = out;
            if i < self.num_layers - 1 && self.dropout > 0.0 {
                x = self.dropout_layer.forward(&x, true)?;
            }
        }
        Ok((x, hidden_states))
    }

    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<MLstmstate>> {
        let cell = &self.layers[0];
        let d_inner = cell.hidden_size * cell.expansion_factor;
        let head_dim = d_inner / cell.num_heads;
        (0..self.num_layers).map(|_| {
            Ok(MLstmstate::new(
                Tensor::zeros((batch_size, cell.num_heads, head_dim, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, cell.hidden_size), DType::F32, device)?,
                Tensor::zeros((batch_size, cell.num_heads, 1, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, cell.num_heads, 1, 1), DType::F32, device)?,
            ))
        }).collect()
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
    pub outnorm: LayerNorm,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub expansion_factor: usize,
}

impl MLstmcell {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, expansion_factor: usize, vb: VarBuilder) -> Result<Self> {
        let d_inner = hidden_size * expansion_factor;
        let head_dim = d_inner / num_heads;

        // Projections initialized with small stdev to prevent immediate saturation
        let w_q = Linear::new(vb.pp("w_q").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);
        let w_k = Linear::new(vb.pp("w_k").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);
        let w_v = Linear::new(vb.pp("w_v").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);

        // Gates initialized according to NXAI/Paper hybrids for maximum stability
        let w_i_spec = vb.pp("w_i");
        let w_i_weights = w_i_spec.get_with_hints((num_heads, 3 * d_inner), "weight", Init::Const(0.0))?;
        let w_i_bias = w_i_spec.get_with_hints(num_heads, "bias", Init::Randn { mean: 0.0, stdev: 0.1 })?;
        let w_i = Linear::new(w_i_weights, Some(w_i_bias));

        let w_f_spec = vb.pp("w_f");
        let w_f_weights = w_f_spec.get_with_hints((num_heads, 3 * d_inner), "weight", Init::Const(0.0))?;
        
        // Linspace bias [3.0, ..., 6.0] as in NXAI. 
        // Note: forward_sequence will clamp this to <= 0 for mLSTM exponential stability.
        let mut f_bias_vec: Vec<f32> = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            let val = 3.0 + (i as f32 * 3.0 / (num_heads as f32 - 1.0).max(1.0));
            f_bias_vec.push(val);
        }
        let w_f_bias = Tensor::from_vec(f_bias_vec, num_heads, vb.device())?;
        let w_f = Linear::new(w_f_weights, Some(w_f_bias));

        // Output gate/branch (SiLU is often preferred in modern xLSTM layers)
        let w_o = linear(input_size, d_inner, vb.pp("w_o"))?;
        let w_down = linear(d_inner, hidden_size, vb.pp("w_down"))?;
        let outnorm = layer_norm(head_dim, 1e-5, vb.pp("outnorm"))?;

        Ok(Self { w_q, w_k, w_v, w_i, w_f, w_o, w_down, outnorm, hidden_size, num_heads, expansion_factor })
    }

    pub fn forward_sequence(&self, input_seq: &Tensor, state: &MLstmstate) -> Result<(Tensor, MLstmstate)> {
        let (batch_size, seq_len, _) = input_seq.dims3()?;
        let d_inner = self.hidden_size * self.expansion_factor;
        let head_dim = d_inner / self.num_heads;
        let device = input_seq.device();

        // 1. Base Projections
        let q_proj = self.w_q.forward(input_seq)?;
        let k_proj = self.w_k.forward(input_seq)?;
        let v_proj = self.w_v.forward(input_seq)?;

        // Reshape to Multi-Head: [B, H, S, D_h]
        let q = q_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // Scale per paper: Q and K scaled so QK^T is scaled by 1/sqrt(head_dim)
        let q = (q / (head_dim as f64).powf(0.25))?;
        let k = (k / (head_dim as f64).powf(0.25))?;

        // 2. Branch Gating (SiLU as in NXAI snippet)
        let o_gate = ops::silu(&self.w_o.forward(input_seq)?)?
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?.contiguous()?;

        // 3. Gates
        let gate_input = Tensor::cat(&[&q_proj, &k_proj, &v_proj], 2)?;
        let i_log = self.w_i.forward(&gate_input)?.permute((0, 2, 1))?.unsqueeze(3)?.contiguous()?;
        let f_log = self.w_f.forward(&gate_input)?.permute((0, 2, 1))?.unsqueeze(3)?.contiguous()?;
        
        // LOG GATE CLAMPING: Relaxed to allow max-stabilization to work.
        let i_log = i_log.clamp(-20.0, 20.0)?;
        let f_log = f_log.clamp(-20.0, 20.0)?;

        // 4. Parallel Kernel (Dual Form)
        let indices = Tensor::arange(0u32, seq_len as u32, device)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?
            .ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?
            .to_dtype(DType::U8)?;

        let f_cumsum = f_log.cumsum(2)?;

        // log_W[t, k] = f_cumsum[t] - f_cumsum[k] + i_log[k]
        let f_t = f_cumsum.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let f_k = f_cumsum.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let i_k = i_log.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;

        let log_weights = f_t.sub(&f_k)?.add(&i_k)?;
        
        // Stabilized masking
        let neg_inf = Tensor::new(-1e10f32, device)?.broadcast_as(log_weights.shape())?;
        let log_weights_masked = mask.broadcast_as(log_weights.shape())?.where_cond(&log_weights, &neg_inf)?;

        // 5. Max-Stabilization (m_t)
        let m_0 = state.max_gate_log.clone();
        let m_prev = f_cumsum.broadcast_add(&m_0)?;
        let m_local = log_weights_masked.max(3)?.unsqueeze(3)?;
        let m_t = m_local.maximum(&m_prev)?;
        
        let weights = log_weights_masked.broadcast_sub(&m_t.clone())?.exp()?;

        // 6. Compute Hidden State
        let qk_t = q.matmul(&k.permute((0, 1, 3, 2))?)?;
        let att = weights.broadcast_mul(&qk_t)?;
        let h_p = att.matmul(&v)?;

        let initial_scale = f_cumsum.broadcast_add(&m_0)?.broadcast_sub(&m_t.clone())?.exp()?;
        let h_init = q.matmul(&state.cell)?;
        let h_total = (h_p + h_init.broadcast_mul(&initial_scale.clone())?)?;

        // 7. Normalizer (n_t) - Paper Eq 19
        let n_p = weights.matmul(&k)?; 
        let n_init = state.normalizer.broadcast_mul(&initial_scale.clone())?;
        let n_total = (n_p + n_init)?; // [B, H, S, D]
        
        let q_dot_n = (n_total.clone() * q.clone())?.sum_keepdim(3)?; // q_t^T n_t
        // Denominator z_t = max(initial_scale, |q_t^T n_t|)
        let denominator = q_dot_n.abs()?.maximum(&initial_scale)?.clamp(1e-12, f32::MAX)?;
        let h_norm = h_total.broadcast_div(&denominator)?;
        
        // Final Output
        let h_permuted = h_norm.permute((0, 2, 1, 3))?.contiguous()?;
        let h_ln = self.outnorm.forward(&h_permuted)?;
        let h_gated = h_ln.broadcast_mul(&o_gate.permute((0, 2, 1, 3))?.contiguous()?)?;
        let out = self.w_down.forward(&h_gated.reshape((batch_size, seq_len, d_inner))?)?;

        // 8. Update State
        let last_idx = seq_len - 1;
        let m_next = m_t.narrow(2, last_idx, 1)?; 
        let n_next = n_total.narrow(2, last_idx, 1)?;
        
        let i_scale_last = initial_scale.narrow(2, last_idx, 1)?;
        let w_last = weights.narrow(2, last_idx, 1)?;
        let v_weighted = v.permute((0, 1, 3, 2))?.broadcast_mul(&w_last)?; 
        let c_upd = v_weighted.matmul(&k)?;
        let c_next = state.cell.broadcast_mul(&i_scale_last)?.add(&c_upd)?;

        let next_state = MLstmstate::new(
            c_next, 
            out.narrow(1, last_idx, 1)?.reshape((batch_size, self.hidden_size))?, 
            n_next, 
            m_next
        );

        Ok((out, next_state))
    }
}
