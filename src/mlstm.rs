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

        // Projections initialized with small stdev as per NXAI
        let w_q = Linear::new(vb.pp("w_q").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);
        let w_k = Linear::new(vb.pp("w_k").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);
        let w_v = Linear::new(vb.pp("w_v").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?, None);

        // Gates Weights initialized to zero as per NXAI snippet
        let w_i_spec = vb.pp("w_i");
        let w_i_weights = w_i_spec.get_with_hints((num_heads, 3 * d_inner), "weight", Init::Const(0.0))?;
        let w_i_bias = w_i_spec.get_with_hints(num_heads, "bias", Init::Randn { mean: 0.0, stdev: 0.1 })?;
        let w_i = Linear::new(w_i_weights, Some(w_i_bias));

        let w_f_spec = vb.pp("w_f");
        let w_f_weights = w_f_spec.get_with_hints((num_heads, 3 * d_inner), "weight", Init::Const(0.0))?;
        let mut f_bias_vec: Vec<f32> = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            let val = 3.0 + (i as f32 * 3.0 / (num_heads as f32 - 1.0).max(1.0));
            f_bias_vec.push(val);
        }
        let w_f_bias = Tensor::from_vec(f_bias_vec, num_heads, vb.device())?;
        let w_f = Linear::new(w_f_weights, Some(w_f_bias));

        // Output gate (Branch z in NXAI mLSTMLayer)
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

        // 1. Projections
        let q_proj = self.w_q.forward(input_seq)?;
        let k_proj = self.w_k.forward(input_seq)?;
        let v_proj = self.w_v.forward(input_seq)?;

        let q = q_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // Scaling per paper Section 3.2
        let q = (q / (head_dim as f64).powf(0.25))?;
        let k = (k / (head_dim as f64).powf(0.25))?;

        // 2. Branch z (Output gating branch with SiLU as in NXAI layer)
        let z = self.w_o.forward(input_seq)?;
        let z_act = ops::silu(&z)?.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // 3. Gates
        let gate_input = Tensor::cat(&[&q_proj, &k_proj, &v_proj], 2)?;
        let i_log = self.w_i.forward(&gate_input)?.reshape((batch_size, seq_len, self.num_heads, 1))?.permute((0, 2, 1, 3))?.contiguous()?;
        let f_log = self.w_f.forward(&gate_input)?.reshape((batch_size, seq_len, self.num_heads, 1))?.permute((0, 2, 1, 3))?.contiguous()?;
        
        let i_log = i_log.clamp(-3.0, 3.0)?;
        let f_log = f_log.clamp(-3.0, 3.0)?;

        let indices = Tensor::arange(0u32, seq_len as u32, device)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?
            .ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?
            .to_dtype(DType::U8)?;

        // 4. Parallel Kernel (Dual Form)
        let f_cumsum = f_log.cumsum(2)?;
        let f_t = f_cumsum.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let f_k = f_cumsum.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let i_k = i_log.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;

        let log_weights = f_t.sub(&f_k)?.add(&i_k)?;
        let neg_inf = Tensor::new(-1e10f32, device)?.broadcast_as(log_weights.shape())?;
        let log_weights_masked = mask.broadcast_as(log_weights.shape())?.where_cond(&log_weights, &neg_inf)?;

        // 5. Max-Stabilization
        let m_0 = state.max_gate_log.clone();
        let m_t = log_weights_masked.max(3)?.unsqueeze(3)?.maximum(&f_cumsum.broadcast_add(&m_0)?)?;
        let weights = log_weights_masked.broadcast_sub(&m_t.clone())?.exp()?;

        // 6. Compute State and Normalizer
        let att = weights.broadcast_mul(&q.matmul(&k.permute((0, 1, 3, 2))?)?)?;
        let h_p = att.matmul(&v)?;

        let initial_scale = f_cumsum.broadcast_add(&m_0)?.broadcast_sub(&m_t.clone())?.exp()?;
        let h_total = (h_p + q.matmul(&state.cell)?.broadcast_mul(&initial_scale.clone())?)?;

        let n_total = (weights.matmul(&k)? + state.normalizer.broadcast_mul(&initial_scale.clone())?)?;
        
        // Denominator z_t = max(exp(F_t + m_0 - m_t), |q_t^T n_t|) as per paper Eq 19
        let q_dot_n = (n_total.clone() * q)?.sum_keepdim(3)?;
        let denominator = q_dot_n.abs()?.maximum(&initial_scale)?.clamp(1e-6, f32::MAX)?;
        let h_norm = h_total.broadcast_div(&denominator)?;
        
        // Final Output
        let h_ln = self.outnorm.forward(&h_norm.permute((0, 2, 1, 3))?.contiguous()?)?;
        let h_gated = h_ln.broadcast_mul(&z_act.permute((0, 2, 1, 3))?.contiguous()?)?;
        let out = self.w_down.forward(&h_gated.reshape((batch_size, seq_len, d_inner))?)?;

        // 7. Update State
        let last_idx = seq_len - 1;
        let i_scale_last = initial_scale.narrow(2, last_idx, 1)?;
        let w_last = weights.narrow(2, last_idx, 1)?;
        let c_upd = v.permute((0, 1, 3, 2))?.broadcast_mul(&w_last)?.matmul(&k)?;
        let mut c_next = state.cell.broadcast_mul(&i_scale_last)?.add(&c_upd)?;
        
        let c_max = c_next.abs()?.max_all()?.to_scalar::<f32>()?;
        if c_max > 10.0 {
            c_next = (c_next * (10.0 / (1.0 + c_max / 1.0)) as f64)?;
        }

        let next_state = MLstmstate::new(
            c_next, 
            out.narrow(1, last_idx, 1)?.reshape((batch_size, self.hidden_size))?, 
            n_total.narrow(2, last_idx, 1)?, 
            m_t.narrow(2, last_idx, 1)?
        );

        Ok((out, next_state))
    }
}
