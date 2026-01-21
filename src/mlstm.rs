/*
# mLSTM: Matrix Long Short-Term Memory
Implementación Dual Form Paralela Estable para Candle.
Fiel al paper xLSTM (Beck et al. 2024) y referencias de NXAI.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, layer_norm, init::Init};

/// Estado del mLSTM para persistencia entre secuencias (Batch-to-Batch).
#[derive(Clone, Debug)]
pub struct MLstmstate {
    /// Memoria Matricial C_t: [B, NH, DH, DH]
    pub cell: Tensor,
    /// Normalizador Vectorial n_t: [B, NH, 1, DH]
    pub normalizer: Tensor,
    /// Max-stabilizer m_t: [B, NH, 1, 1]
    pub max_gate_log: Tensor,
    /// Placeholder para consistencia: [B, D_hidden]
    pub hidden: Tensor,
}

impl MLstmstate {
    pub fn new(cell: Tensor, normalizer: Tensor, max_gate_log: Tensor, hidden: Tensor) -> Self {
        Self { cell, normalizer, max_gate_log, hidden }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
            hidden: self.hidden.detach(),
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
                Tensor::zeros((batch_size, cell.num_heads, 1, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, cell.num_heads, 1, 1), DType::F32, device)?,
                Tensor::zeros((batch_size, cell.hidden_size), DType::F32, device)?,
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

        // Kaiming Normal manual
        let k_std = (1.0 / input_size as f64).sqrt();
        let w_q = Linear::new(vb.pp("w_q").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: k_std })?, None);
        let w_k = Linear::new(vb.pp("w_k").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: k_std })?, None);
        let w_v = Linear::new(vb.pp("w_v").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: k_std })?, None);

        // Puertas directamente desde el input (Gradiente directo)
        let w_i_spec = vb.pp("w_i");
        let w_i = Linear::new(w_i_spec.get_with_hints((num_heads, input_size), "weight", Init::Const(0.0))?, Some(w_i_spec.get_with_hints(num_heads, "bias", Init::Randn { mean: 0.0, stdev: 0.1 })?));

        let w_f_spec = vb.pp("w_f");
        let mut f_bias_vec: Vec<f32> = Vec::with_capacity(num_heads);
        for i in 0..num_heads { f_bias_vec.push(3.0 + (i as f32 * 3.0 / (num_heads as f32 - 1.0).max(1.0))); }
        let w_f = Linear::new(w_f_spec.get_with_hints((num_heads, input_size), "weight", Init::Const(0.0))?, Some(Tensor::from_vec(f_bias_vec, num_heads, vb.device())?));

        // Proyecciones finales suaves para facilitar el flujo residual
        let w_o = Linear::new(vb.pp("w_o").get_with_hints((d_inner, input_size), "weight", Init::Randn { mean: 0.0, stdev: 0.01 })?, Some(vb.pp("w_o").get_with_hints(d_inner, "bias", Init::Const(0.0))?));
        let w_down = Linear::new(vb.pp("w_down").get_with_hints((hidden_size, d_inner), "weight", Init::Randn { mean: 0.0, stdev: 0.01 })?, Some(vb.pp("w_down").get_with_hints(hidden_size, "bias", Init::Const(0.0))?));
        let outnorm = layer_norm(head_dim, 1e-5, vb.pp("outnorm"))?;

        Ok(Self { w_q, w_k, w_v, w_i, w_f, w_o, w_down, outnorm, hidden_size, num_heads, expansion_factor })
    }

    pub fn forward_sequence(&self, input_seq: &Tensor, state: &MLstmstate) -> Result<(Tensor, MLstmstate)> {
        let (batch_size, seq_len, _) = input_seq.dims3()?;
        let d_inner = self.hidden_size * self.expansion_factor;
        let head_dim = d_inner / self.num_heads;
        let device = input_seq.device();

        // 1. Proyecciones Q, K, V
        let q_proj = self.w_q.forward(input_seq)?;
        let k_proj = self.w_k.forward(input_seq)?;
        let v_proj = self.w_v.forward(input_seq)?;

        let q = q_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v_proj.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // Escalado dh^-1/4 (Paper v2)
        let scale = (head_dim as f64).powf(0.25);
        let q = (q / scale)?;
        let k = (k / scale)?;
        
        // 2. Puertas (Gradiente directo del input)
        let i_log = self.w_i.forward(input_seq)?.reshape((batch_size, seq_len, self.num_heads, 1))?.permute((0, 2, 1, 3))?.clamp(-6.0, 6.0)?;
        let f_log = self.w_f.forward(input_seq)?.reshape((batch_size, seq_len, self.num_heads, 1))?.permute((0, 2, 1, 3))?.clamp(-6.0, 0.0)?;
        let o_gate = ops::silu(&self.w_o.forward(input_seq)?)?.reshape((batch_size, seq_len, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // 3. Parallel Kernel (Dual Form)
        let f_cumsum = f_log.cumsum(2)?;
        let m_0 = state.max_gate_log.clone();
        
        // W[t, k] = f_cumsum[t] - f_cumsum[k] + i_log[k]
        let f_t = f_cumsum.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let f_k = f_cumsum.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let i_k = i_log.permute((0, 1, 3, 2))?.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        let log_weight_gates = f_t.sub(&f_k)?.add(&i_k)?;
        
        // Causal Masking
        let indices = Tensor::arange(0u32, seq_len as u32, device)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?.ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?.to_dtype(DType::U8)?;
        let neg_inf = Tensor::new(-1e10f32, device)?.broadcast_as(log_weight_gates.shape())?;
        let log_weight_masked = mask.broadcast_as(log_weight_gates.shape())?.where_cond(&log_weight_gates, &neg_inf)?;

        // Max-Stabilization m_t
        let m_prev = f_cumsum.broadcast_add(&m_0)?;
        let m_local = log_weight_masked.max(3)?.unsqueeze(3)?;
        let m_t = m_local.maximum(&m_prev.broadcast_as(m_local.shape())?)?;
        let weights = log_weight_masked.broadcast_sub(&m_t.clone())?.exp()?;

        // h_p = (weights * (Q @ K^T)) @ V
        let qk_t = q.matmul(&k.permute((0, 1, 3, 2))?)?;
        let h_p = weights.broadcast_mul(&qk_t)?.matmul(&v)?;

        // Contribución de memoria anterior
        let initial_scale = f_cumsum.broadcast_add(&m_0)?.broadcast_sub(&m_t.clone())?.exp()?;
        let h_init = q.matmul(&state.cell)?;
        let h_tilde = (h_p + h_init.broadcast_mul(&initial_scale.clone())?)?;

        // 4. Normalizador Eq 19 (Denominador z_t)
        let n_p = weights.matmul(&k)?; 
        let n_init = state.normalizer.broadcast_mul(&initial_scale.clone())?;
        let n_total = (n_p + n_init)?; 
        
        let q_dot_n = (q.clone() * n_total.clone())?.sum_keepdim(3)?;
        let denominator = q_dot_n.abs()?.clamp(1e-6, f32::MAX)?;
        let h_norm = h_tilde.broadcast_div(&denominator)?;
        
        // Output y Post-Gating
        let h_ln = self.outnorm.forward(&h_norm.permute((0, 2, 1, 3))?.contiguous()?)?;
        let h_gated = h_ln.broadcast_mul(&o_gate.permute((0, 2, 1, 3))?.contiguous()?)?;
        let out = self.w_down.forward(&h_gated.reshape((batch_size, seq_len, d_inner))?)?;

        // 5. Update State (Batch-to-Batch persistence)
        let last_idx = seq_len - 1;
        let m_next = m_t.narrow(2, last_idx, 1)?;
        let n_next = n_total.narrow(2, last_idx, 1)?;
        
        // C_next = i_scale * C_prev + (v_last @ k_last^T)
        // Aplicamos el input gate a la actualización
        let i_scale_last = initial_scale.narrow(2, last_idx, 1)?;
        let v_last = v.narrow(2, last_idx, 1)?.transpose(2, 3)?;
        let k_last = k.narrow(2, last_idx, 1)?;
        let i_last = i_log.narrow(2, last_idx, 1)?.exp()?;
        let c_upd = v_last.matmul(&k_last.broadcast_mul(&i_last)?)?;
        let c_next = state.cell.broadcast_mul(&i_scale_last)?.add(&c_upd)?;

        // Captura del hidden state persistente (último elemento)
        let hidden_persist = out.narrow(1, last_idx, 1)?.reshape((batch_size, self.hidden_size))?;

        Ok((out, MLstmstate::new(c_next, n_next, m_next, hidden_persist)))
    }
}
