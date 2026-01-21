/*
# mLSTM: Matrix Long Short-Term Memory
Implementación exacta según: "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517v2)
Sección 2.3 y Apéndice A.3. Sin normalizaciones redundantes para permitir el aprendizaje agresivo.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, ops, linear};

/// Estado para mLSTM (Matrix Memory) - C_t, n_t y m_t
#[derive(Clone, Debug)]
pub struct MLstmstate {
    pub cell: Tensor,       // matrix memory
    pub normalizer: Tensor, // normalizer state
    pub m_t: Tensor,        // stabilizer state
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
            dropout: Dropout::new(self.dropout),
            num_layers: self.num_layers,
        })
    }
}

#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout: Dropout,
    pub num_layers: usize,
}

impl MLstm {
    pub fn forward(&self, input: &Tensor, states: Option<Vec<MLstmstate>>) -> Result<(Tensor, Vec<MLstmstate>)> {
        let mut current_input = input.clone();
        let mut new_states = Vec::with_capacity(self.num_layers);
        
        for (i, layer) in self.layers.iter().enumerate() {
            let state_ref = states.as_ref().map(|s| &s[i]);
            let (output, next_state) = layer.forward_sequence(&current_input, state_ref)?;
            current_input = self.dropout.forward(&output, true)?;
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

    /// Parallel Dual Form stable (arXiv:2405.04517v2 Appendix A.3)
    pub fn forward_sequence(&self, x: &Tensor, _state_prev: Option<&MLstmstate>) -> Result<(Tensor, MLstmstate)> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let dev = x.device();

        // 1. Proyecciones
        let q = self.w_q.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = self.w_k.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = self.w_v.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        
        // Log-space pre-activaciones (para puertas exponenciales disruptivas)
        let log_i = self.w_i.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let log_f = self.w_f.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let o_gate = ops::sigmoid(&self.w_o.forward(x)?)?;

        // Promediamos sobre la dimensión de la cabeza para obtener un escalar por paso
        let log_i_s = log_i.mean(3)?;
        let log_f_s = log_f.mean(3)?;
        
        // 2. Acumulación causal de olvido (Ecuación 101)
        let s = log_f_s.cumsum(2)?; 
        
        // log_D[i, j] = log_i[j] + s[i] - s[j]
        let log_d = s.unsqueeze(3)? 
            .broadcast_sub(&s.unsqueeze(2)?)? 
            .broadcast_add(&log_i_s.unsqueeze(2)?)?; 

        // Máscara Causal (U8)
        let indices = Tensor::arange(0u32, seq_len as u32, dev)?;
        let mask = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?
            .ge(&indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?)?
            .unsqueeze(0)?.unsqueeze(0)?;
            
        let neg_inf = Tensor::new(-1e10f32, dev)?.broadcast_as(log_d.shape())?;
        let log_d_masked = mask.broadcast_as(log_d.shape())?.where_cond(&log_d, &neg_inf)?;

        // Stabilizer m_t (Eq. 80-81) para evitar overflow sin sacrificar potencia
        let m = log_d_masked.max_keepdim(3)?; 
        let d_prime = log_d_masked.broadcast_sub(&m)?.exp()?; 

        // 3. Matrix Memory Retrieval (Eq. 82-86)
        let scale = Tensor::new((self.head_dim as f32).sqrt(), dev)?;
        let q_scaled = q.broadcast_div(&scale)?;
        let qk_t = q_scaled.matmul(&k.transpose(2, 3)?)?;
        
        let matrix_weights = qk_t.broadcast_mul(&d_prime)?;
        let h_raw = matrix_weights.matmul(&v)?; 

        // Normalizador de Matrix Memory (Eq. 84) - Única normalización obligatoria
        let b = matrix_weights.sum_keepdim(3)?;
        let n = b.abs()?.maximum(&(m.neg()?.exp()?))?;
        
        // El hidden state recuperado de la memoria
        let h_norm = h_raw.broadcast_div(&n)?; 

        // 4. Salida Proyectada
        let h_reshaped = h_norm.permute((0, 2, 1, 3))?.reshape((b_sz, seq_len, ()))?;
        let out = self.w_down.forward(&(o_gate * h_reshaped)?)?;

        // Estado final para inferencia recurrente
        let final_state = MLstmstate::new(
            Tensor::zeros((b_sz, self.num_heads, self.head_dim, self.head_dim), DType::F32, dev)?,
            Tensor::zeros((b_sz, self.num_heads, self.head_dim), DType::F32, dev)?,
            m.narrow(2, seq_len - 1, 1)?.squeeze(3)?
        );

        Ok((out, final_state))
    }
}
