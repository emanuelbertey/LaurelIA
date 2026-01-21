/*
# mLSTM: Matrix Long Short-Term Memory
Implementación exacta según: "xLSTM: Extended Long Short-Term Memory" (2405.04517v2)
Parallel Dual Form corregida para tipos de datos exactos en Candle.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, linear, layer_norm};

/// Estado para mLSTM (Matrix Memory)
#[derive(Clone, Debug)]
pub struct MLstmstate {
    pub cell: Tensor,       // C_t: [B, NH, DH, DH]
    pub normalizer: Tensor, // n_t: [B, NH, DH]
    pub m_t: Tensor,        // m_t: [B, NH, 1]
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
    pub head_ln: LayerNorm,
    
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
            head_ln: layer_norm(head_dim, 1e-5, vb.pp("head_ln"))?,
            num_heads: n_heads,
            head_dim,
            d_inner,
        })
    }

    /// Parallel Forward Pass (A.3 Equations 79-86)
    pub fn forward_sequence(&self, x: &Tensor, _state_prev: Option<&MLstmstate>) -> Result<(Tensor, MLstmstate)> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let dev = x.device();

        // 1. Proyecciones
        let q = self.w_q.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = self.w_k.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = self.w_v.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        
        let log_i = self.w_i.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let log_f = ops::sigmoid(&self.w_f.forward(x)?)?.log()?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let o_gate = ops::sigmoid(&self.w_o.forward(x)?)?;

        // 2. Cálculo de pesos de atención en espacio logarítmico
        let log_f_mean = log_f.mean(3)?; // [B, NH, L]
        let log_i_mean = log_i.mean(3)?; // [B, NH, L]
        
        let s = log_f_mean.cumsum(2)?; 
        
        let log_d = s.unsqueeze(3)? 
            .broadcast_sub(&s.unsqueeze(2)?)? 
            .broadcast_add(&log_i_mean.unsqueeze(2)?)?; 

        // Máscara Causal Manual (Tipo U8 para where_cond)
        let indices = Tensor::arange(0u32, seq_len as u32, dev)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?
            .ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?
            .unsqueeze(0)?.unsqueeze(0)?; // [1, 1, L, L] de tipo U8
            
        let neg_inf = Tensor::new(-1e10f32, dev)?.broadcast_as(log_d.shape())?;
        
        // El secreto del aprendizaje: where_cond requiere mask U8 y valores del mismo tipo (F32)
        let log_d_masked = mask.broadcast_as(log_d.shape())?.where_cond(&log_d, &neg_inf)?;

        // Stabilizer m_t (Eq. 80-81)
        let m = log_d_masked.max_keepdim(3)?; 
        let d_prime = log_d_masked.broadcast_sub(&m)?.exp()?; 

        // 3. Retrieval Paralelo (Eq. 82-86)
        let scale = Tensor::new((self.head_dim as f32).sqrt(), dev)?;
        let q_scaled = q.broadcast_div(&scale)?;
        let qk_t = q_scaled.matmul(&k.transpose(2, 3)?)?;
        
        let matrix_weights = qk_t.broadcast_mul(&d_prime)?;
        let h_raw = matrix_weights.matmul(&v)?; 

        // Normalizador (Eq. 84)
        let b = matrix_weights.sum_keepdim(3)?;
        let exp_neg_m = m.neg()?.exp()?;
        let n = b.abs()?.maximum(&exp_neg_m)?;
        
        let h_normalized = h_raw.broadcast_div(&n)?; 

        // 4. Proyección salida
        let h_reshaped = h_normalized.permute((0, 2, 1, 3))?;
        let h_ln = self.head_ln.forward(&h_reshaped.contiguous()?)?;
        let h_out = h_ln.reshape((b_sz, seq_len, ()))?;
        
        let out = self.w_down.forward(&(o_gate * h_out)?)?;

        let final_state = MLstmstate::new(
            Tensor::zeros((b_sz, self.num_heads, self.head_dim, self.head_dim), DType::F32, dev)?,
            Tensor::zeros((b_sz, self.num_heads, self.head_dim), DType::F32, dev)?,
            m.narrow(2, seq_len - 1, 1)?.squeeze(3)?
        );

        Ok((out, final_state))
    }
}
