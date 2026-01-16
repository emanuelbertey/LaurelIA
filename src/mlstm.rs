/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, linear_no_bias, layer_norm};

/// State for mLSTM containing cell matrix and hidden state
#[derive(Clone, Debug)]
pub struct MLstmstate {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor,
    /// Normalizer state - vector of shape [`batch_size`, `num_heads`, `head_dim`]
    pub normalizer: Tensor,
    /// Global max gate state for numeric stability - shape [`batch_size`, `num_heads`, 1]
    pub max_gate_log: Tensor,
}

impl MLstmstate {
    /// Create a new mLSTM state
    pub fn new(
        cell: Tensor,
        hidden: Tensor,
        normalizer: Tensor,
        max_gate_log: Tensor,
    ) -> Self {
        Self {
            cell,
            hidden,
            normalizer,
            max_gate_log,
        }
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

/// Configuration for mLSTM
#[derive(Debug, Clone)]
pub struct MLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    // Initializer is handled by VarBuilder/init logic
}

impl MLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            num_heads,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize a new mLSTM
    pub fn init(&self, vb: VarBuilder) -> Result<MLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i));
            layers.push(MLstmcell::new(input_size, self.d_hidden, self.num_heads, layer_vb)?);
        }

        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
        })
    }
}

/// mLSTM layer implementation
#[derive(Debug)]
pub struct MLstm {
    /// Stack of mLSTM cells
    pub layers: Vec<MLstmcell>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl MLstm {
    /// Forward pass through mLSTM consuming and returning states
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<MLstmstate>>,
    ) -> Result<(Tensor, Vec<MLstmstate>)> {
        let (batch_size, _seq_length, _) = input_seq.dims3()?;
        let device = input_seq.device();

        // Inicializar estados
        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, device)?,
        };
        
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // mLSTM processes the entire sequence using the parallel kernel (Dual Form)
            let old_state = &hidden_states[layer_idx];
            
            // We pass the full sequence
            let (h_seq, new_state) = layer.forward_sequence(&layer_input, old_state)?;
            
            // Store the final state for future sequences (re-injecting continuity)
            hidden_states[layer_idx] = new_state;

            // Inter-layer Dropout
            layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(&h_seq, false)?
            } else {
                h_seq
            };
        }

        Ok((layer_input, hidden_states))
    }

    /// Initialize hidden states
    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<MLstmstate>> {
        let head_dim = self.d_hidden / self.num_heads;
        
        (0..self.num_layers)
            .map(|_| {
                Ok(MLstmstate::new(
                    Tensor::zeros((batch_size, self.num_heads, head_dim, head_dim), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.num_heads, head_dim), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.num_heads, 1), DType::F32, device)?,
                ))
            })
            .collect()
    }
}

/// mLSTM cell implementation with matrix memory
#[derive(Debug)]
pub struct MLstmcell {
    /// Weight matrix for input to gates
    pub weight_ih: Tensor,
    /// Weight matrix for hidden to gates
    pub weight_hh: Tensor,
    /// Bias for gates (trainable base)
    pub bias: Tensor,
    /// Fixed bias offset to match Burn implementation (e.g. forget gate bias 0.5)
    pub bias_offset: Tensor,
    /// Query projection
    pub w_q: Linear,
    /// Key projection
    pub w_k: Linear,
    /// Value projection
    pub w_v: Linear,
    /// LayerNorm for multi-head normalization
    pub ln: LayerNorm,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
}

impl MLstmcell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight_ih = vb.get_with_hints(
            (3 * hidden_size, input_size), 
            "weight_ih", 
            candle_nn::init::DEFAULT_KAIMING_NORMAL 
        )?;
        
        // weight_hh maintained for compatibility/sequential mode if needed
        let weight_hh = vb.get_with_hints(
            (3 * hidden_size, hidden_size), 
            "weight_hh", 
            candle_nn::init::DEFAULT_KAIMING_NORMAL
        )?;

        let bias = vb.get_with_hints(
            3 * hidden_size, 
            "bias", 
            candle_nn::init::Init::Const(0.0)
        )?;

        // Bias offset to replicate Burn's manual initialization:
        // forget gate bias = 0.5, others = 0.0
        let device = vb.device();
        let mut b_offset_vals = vec![0.0f32; 3 * hidden_size];
        for i in hidden_size..(2 * hidden_size) {
            b_offset_vals[i] = 0.5;
        }
        let bias_offset = Tensor::from_vec(b_offset_vals, (3 * hidden_size,), device)?;

        let head_dim = hidden_size / num_heads;

        // Q, K, V
        let w_q = linear_no_bias(input_size, hidden_size, vb.pp("w_q"))?;
        let w_k = linear_no_bias(input_size, hidden_size, vb.pp("w_k"))?;
        let w_v = linear_no_bias(input_size, hidden_size, vb.pp("w_v"))?;
        
        let ln = layer_norm(head_dim, 1e-5, vb.pp("ln"))?;

        Ok(Self {
            weight_ih,
            weight_hh,
            bias,
            bias_offset,
            w_q,
            w_k,
            w_v,
            ln,
            input_size,
            hidden_size,
            num_heads,
        })
    }

    /// Forward pass through mLSTM cell consuming the state
    pub fn forward_sequence(
        &self,
        input_seq: &Tensor,
        state: &MLstmstate,
    ) -> Result<(Tensor, MLstmstate)> {
        let (batch_size, seq_len, _) = input_seq.dims3()?;
        let head_dim = self.hidden_size / self.num_heads;
        let device = input_seq.device();

        // 1. Parallel Projections (Q, K, V)
        // input_seq: [B, S, D_in]
        // w_q(input): [B, S, D_h] -> reshape [B, S, H, D_head] -> transpose(1,2) -> [B, H, S, D_head]
        let q = self.w_q.forward(input_seq)?
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))? // [B, H, S, D_h]
            .contiguous()?;
        let k = self.w_k.forward(input_seq)?
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = self.w_v.forward(input_seq)?
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let scale = (head_dim as f64).sqrt();
        let q = (q / scale)?;
        let k = (k / scale)?;

        // 2. Parallel Gates
        // weight_ih: [3*D_h, D_in]. input: [B, S, D_in].
        // input @ weight_ih^T = [B, S, 3*D_h]
        // Flatten input for matmul: [B, S, D] -> [B*S, D]
        let (batch_size, seq_len, _d_in) = input_seq.dims3()?;
        let input_flat = input_seq.reshape((batch_size * seq_len, self.input_size))?;
        
        let weight_ih_t = self.weight_ih.t()?.contiguous()?;
        let gates_flat = input_flat.matmul(&weight_ih_t)?;
        let gates = gates_flat.reshape((batch_size, seq_len, 3 * self.hidden_size))?
            .broadcast_add(&self.bias.reshape((1, 1, 3 * self.hidden_size))?)?
            .broadcast_add(&self.bias_offset.reshape((1, 1, 3 * self.hidden_size))?)?;
        
        let chunks = gates.chunk(3, 2)?; // Chunk on last dim (dim 2)
        
        let i_log = chunks[0].clone()
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .clamp(-6.0, 6.0)?;

        let f_log = chunks[1].clone()
            .reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .clamp(-6.0, 6.0)?;
        let f_log = f_log.affine(1.0, 1.0)?;

        let o_pre = chunks[2].clone();
        let o = ops::sigmoid(&o_pre)?; // [B, S, D_hidden]

        let i_log_m = i_log.mean(3)?; // [B, H, S] (keep dim? Burn mean_dim keeps dim? No, usually reduces. Need to check strictly)
        // Burn: mean_dim(3) returns [B, H, S, 1]. Candle mean returns reduced. We need unsqueeze.
        let i_log_m = i_log_m.unsqueeze(3)?;
        let f_log_m = f_log.mean(3)?.unsqueeze(3)?; // [B, H, S, 1]

        // 3. Dual Form (Parallel Kernel)
        // Create causal mask
        // indices: [0..seq_len]
        let indices = Tensor::arange(0u32, seq_len as u32, device)?; 
        let row_idx = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?;
        let col_idx = indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?;
        let mask_tri = row_idx.ge(&col_idx)?.to_dtype(DType::F32)?; // 1 if row>=col (lower triangular)

        // f_cumsum = mask_tri @ f_log_m
        // mask_tri: [S, S]. f_log_m: [B, H, S, 1].
        // We broadcast mask_tri to [1, 1, S, S] or similar.
        // f_log_m is treated as a batch of vectors?
        // Wait, f_cumsum in original: mask_tri [1,1,S,S] matmul f_log_m [B,H,S,1] -> [B,H,S,1] ?
        // [S, S] @ [S, 1] -> [S, 1]. Yes.
        // So broadcast mask_tri. Or simply matmul.
        // Candle matmul supports broadcasting? Yes.
        let f_cumsum = mask_tri.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?
            .matmul(&f_log_m)?; // [B, H, S, 1]

        // log_weights calculation
        // f_cumsum: [B, H, S, 1]. 
        // We need log_weights over [B, H, S(target), S(source)].
        // log_weights = f_cumsum - f_cumsum.transpose + i_log_m.transpose
        // f_cumsum is per time step t (sum 0..t f).
        // For attention: sum_{k=1..t} 
        // Logic seems: w_{t,k} = exp( F_t - F_k + i_k )  (simplified)
        // Original: f_cumsum.clone() - f_cumsum.clone().swap_dims(2, 3) + i_log_m.clone().swap_dims(2, 3)
        // But f_cumsum is [B,H,S,1]. swap_dims(2,3) -> [B,H,1,S].
        // [B,H,S,1] - [B,H,1,S] -> [B,H,S,S] (via broadcast)
        // + [B,H,1,S] (i_log_m transposed) -> [B,H,S,S].
        let f_cumsum_t = f_cumsum.permute((0, 1, 3, 2))?; // [B, H, 1, S]
        let i_log_m_t = i_log_m.permute((0, 1, 3, 2))?;
        let log_weights = f_cumsum.broadcast_sub(&f_cumsum_t)?
            .broadcast_add(&i_log_m_t)?; // [B, H, S, S]

        // Mask future
        let mask_bool = mask_tri.to_dtype(DType::U8)?; // 1s and 0s
        // mask_tri is 1 for valid. 0 for invalid (future).
        // We want to fill INVALID (0) with -inf.
        // log_weights = where(mask_tri == 1, log_weights, -inf)
        // mask_bool is broadcast to [B, H, S, S]? Yes.
        let min_val = -1e10f32;
        // Candle: where_cond(mask (bool), on_true, on_false)
        // Using broadcast mask.
        let log_weights_masked = mask_bool.broadcast_as(log_weights.shape())?
            .where_cond(&log_weights, &Tensor::new(min_val, device)?.broadcast_as(log_weights.shape())?)?;

        // Global Max Stabilization (m_t)
        // m_0: [B, H, 1, 1].
        let m_0 = state.max_gate_log.reshape((batch_size, self.num_heads, 1, 1))?;
        let m_initial = f_cumsum.broadcast_add(&m_0)?; // [B, H, S, 1] + [B, H, 1, 1] using broadcast
        
        // m_i_row = max over dim 3 (source time). [B, H, S, 1]
        let m_i_row = log_weights_masked.max(3)?.unsqueeze(3)?;
        // m_i = max(m_i_row, m_initial)
        let m_i = m_i_row.maximum(&m_initial)?;

        let m_i_stable = m_i.clamp(-10.0, 10.0)?;
        
        let log_diff = log_weights_masked.broadcast_sub(&m_i_stable)?;
        let weights = log_diff.clamp(-20.0, 0.0)?.exp()?;

        // Parallel Hidden State
        // q: [B, H, S, D]. k: [B, H, S, D].
        // k.transpose -> [B, H, D, S].
        // q @ k^T -> [B, H, S, S].
        let kt = k.permute((0, 1, 3, 2))?.contiguous()?; 
        let q_k_t = q.matmul(&kt)?; 
        
        // weights * q_k_t -> elementwise.
        // USO DE BROADCAST_MUL (Evita el error de mismatch) 
        // En lugar de: let w_q_k_t = (&weights * q_k_t)?; 
        let w_q_k_t = weights.broadcast_mul(&q_k_t)?; // [B, H, S, S]
        
        // @ v -> [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        let h_parallel = w_q_k_t.matmul(&v.contiguous()?)?;

        // Initial State Contribution
        // initial_scale = exp(f_cumsum + m_0 - m_i_stable)
        let initial_scale = f_cumsum
            .broadcast_add(&m_0)?
            .broadcast_sub(&m_i_stable)?
            .exp()?
            .clamp(0.0, 1e10)?; // [B, H, S, 1]
        // cell: [B, H, D, D]. q: [B, H, S, D].
        // q @ cell -> [B, H, S, D]
        let h_initial_base = q.matmul(&state.cell)?;
        let h_initial = h_initial_base.broadcast_mul(&initial_scale)?;

        let h_heads = (h_parallel + h_initial)?;

        // Normalizer
        // n_parallel = weights @ k. [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        let n_parallel = weights.matmul(&k.contiguous()?)?;
        // n_initial = normalizer * initial_scale
        // normalizer: [B, H, D] -> [B, H, 1, D] broadcast to [S]
        let n_initial_base = state.normalizer.reshape((batch_size, self.num_heads, 1, head_dim))?
            .broadcast_as((batch_size, self.num_heads, seq_len, head_dim))?;
        let n_initial = n_initial_base.broadcast_mul(&initial_scale)?;
        
        let n_heads = (n_parallel + n_initial)?.clamp(1e-6, f32::MAX)?; 
        let h_normalized = (h_heads / n_heads.clone())?;

        // Layernorm expects [B, S, D] or similar last dim.
        // h_normalized: [B, H, S, D_h].
        // Swap to [B, S, H, D_h] then reshape?
        // Original LayerNorm forward expects one vector? No, Layernorm works on last dim.
        // But here we want elementwise or per head?
        // The original Burn code: swap_dims(1,2) -> [B, S, H, D]. ln.forward(h_reshaped).
        let h_reshaped = h_normalized.permute((0, 2, 1, 3))?.contiguous()?; // [B, S, H, D_h]
        let h_ln = self.ln.forward(&h_reshaped)?; // Works if LN dim matches D_h?
        // Wait, LN config is `head_dim`. So it normalizes the last dimension. Correct.
       
        let h_combined = h_ln.reshape((batch_size, seq_len, self.hidden_size))?;
        let h_seq = ((o * h_combined)?.clamp(-10.0, 10.0))?;

        // 5. Update State (Final T) 
        let last_idx = seq_len - 1; 
        
        // Usamos reshape para asegurar que no se pierdan dimensiones de batch/head 
        let final_m = m_i_stable.narrow(2, last_idx, 1)?.reshape((batch_size, self.num_heads, 1))?; 
        let final_norm = n_heads.narrow(2, last_idx, 1)?.reshape((batch_size, self.num_heads, head_dim))?; 

        let last_initial_scale = initial_scale.narrow(2, last_idx, 1)?.reshape((batch_size, self.num_heads, 1, 1))?; 
        let final_cell_initial = state.cell.broadcast_mul(&last_initial_scale)?; 
        
        let last_row_weights = weights.narrow(2, last_idx, 1)?; // [B, H, 1, S] 
        
        // ACA ESTABA EL ERROR: Usamos broadcast_mul en vez de * 
        let weight_permuted = last_row_weights.permute((0, 1, 3, 2))?.contiguous()?; // [B, H, S, 1] 
        let v_weighted = v.broadcast_mul(&weight_permuted)?; // [B, H, S, D] 
        
        // Matmul final para actualizar la matriz de memoria 
        let v_weighted_t = v_weighted.permute((0, 1, 3, 2))?.contiguous()?; // [B, H, D, S] 
        let final_cell_update = v_weighted_t.matmul(&k.contiguous()?)?; // [B, H, D, D] 

        let mut final_cell = (final_cell_initial + final_cell_update)?; 

        // Soft-normalización dinámica de la matriz de memoria (local por cabeza y muestra)
        // Calculamos |C|_max por cada (batch, head) y aplicamos la escala localmente.
        let abs_cell = final_cell.abs()?;
        let c_max_cols = abs_cell.max(3)?; // Max across head_dim_2 -> [B, H, D]
        let c_max_rows = c_max_cols.max(2)?; // Max across head_dim_1 -> [B, H]
        
        let ten = Tensor::new(10.0f32, device)?;
        let denom = c_max_rows.broadcast_add(&ten)?; 
        let scale_factor = ten.broadcast_div(&denom)?;
        
        // Redimensionar scale_factor de [B, H] a [B, H, 1, 1] para el matmul estable
        let scale_factor = scale_factor.unsqueeze(2)?.unsqueeze(3)?;
        
        final_cell = final_cell.broadcast_mul(&scale_factor)?;
 
        let final_hidden = h_seq.narrow(1, last_idx, 1)?.reshape((batch_size, self.hidden_size))?; 
 
        Ok((h_seq, MLstmstate::new(final_cell, final_hidden, final_norm, final_m)))
    }
}
