/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
use num_traits::{FromPrimitive, ToPrimitive};

/// State for mLSTM containing cell matrix and hidden state
#[derive(Clone, Debug)]
pub struct MLstmstate<B: Backend> {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor<B, 4>,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor<B, 2>,
    /// Normalizer state - vector of shape [`batch_size`, `num_heads`, `head_dim`]
    pub normalizer: Tensor<B, 3>,
    /// Global max gate state for numeric stability - shape [`batch_size`, `num_heads`, 1]
    pub max_gate_log: Tensor<B, 3>,
}

impl<B: Backend> MLstmstate<B> {
    /// Create a new mLSTM state
    pub const fn new(
        cell: Tensor<B, 4>,
        hidden: Tensor<B, 2>,
        normalizer: Tensor<B, 3>,
        max_gate_log: Tensor<B, 3>,
    ) -> Self {
        Self {
            cell,
            hidden,
            normalizer,
            max_gate_log,
        }
    }
}

/// Configuration for mLSTM
#[derive(Config, Debug)]
pub struct MLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    #[config(default = "4")]
    pub num_heads: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}
impl MLstmconfig {
    /// Initialize a new mLSTM
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MLstmcell::new(input_size, self.d_hidden, self.num_heads, &self.initializer, device)
            })
            .collect();

        MLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
        }
    }
}

/// mLSTM layer implementation
#[derive(Module, Debug)]
pub struct MLstm<B: Backend> {
    /// Stack of mLSTM cells
    pub layers: alloc::vec::Vec<MLstmcell<B>>,
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
    pub dropout: f64,
}

impl<B: Backend> MLstm<B> {
    /// Forward pass through mLSTM consuming and returning states
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor of shape [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - States to consume (will be moved)
    ///
    /// # Returns
    /// * Output tensor of shape [`batch_size`, `seq_length`, `hidden_size`]
    /// * New states
    pub fn forward(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MLstmstate<B>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive,
    {
        let device = input_seq.device();
        let [batch_size, _seq_length, _] = input_seq.dims();

        // Inicializar estados
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // mLSTM processes the entire sequence using the parallel kernel (Dual Form)
            let old_state = hidden_states[layer_idx].clone();
            
            let (h_seq, new_state) = layer.forward_sequence(&layer_input, old_state);
            
            // Store the final state for future sequences (re-injecting continuity)
            hidden_states[layer_idx] = new_state;

            // Inter-layer Dropout
            layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(h_seq)
            } else {
                h_seq
            };
        }

        (layer_input, hidden_states)
    }

    /// Initialize hidden states
    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<MLstmstate<B>> {
        let head_dim = self.d_hidden / self.num_heads;
        
        (0..self.num_layers)
            .map(|_| {
                MLstmstate::new(
                    Tensor::zeros([batch_size, self.num_heads, head_dim, head_dim], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.num_heads, head_dim], device),
                    Tensor::zeros([batch_size, self.num_heads, 1], device),
                )
            })
            .collect()
    }
}

/// mLSTM cell implementation with matrix memory
#[derive(Module, Debug)]
pub struct MLstmcell<B: Backend> {
    /// Weight matrix for input to gates
    pub weight_ih: Param<Tensor<B, 2>>,
    /// Weight matrix for hidden to gates
    pub weight_hh: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Query projection
    pub w_q: Linear<B>,
    /// Key projection
    pub w_k: Linear<B>,
    /// Value projection
    pub w_v: Linear<B>,
    /// LayerNorm for multi-head normalization
    pub ln: LayerNorm<B>,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
}

impl<B: Backend> MLstmcell<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        initializer: &Initializer,
        device: &B::Device,
    ) -> Self {
        let weight_ih = initializer.init_with(
            [3 * hidden_size, input_size],
            Some(input_size),
            Some(3 * hidden_size),
            device,
        );
        // weight_hh se mantiene para compatibilidad con predict_last, 
        // pero el kernel paralelo lo ignora para permitir O(1) en tiempo.
        let weight_hh = initializer.init_with(
            [3 * hidden_size, hidden_size],
            Some(hidden_size),
            Some(3 * hidden_size),
            device,
        );

        let mut bias_data = alloc::vec![0.0; 3 * hidden_size];
        for i in 0..hidden_size {
            bias_data[i] = 0.0; // Input gate bias
        }
        for i in hidden_size..(2 * hidden_size) {
            bias_data[i] = 0.5; // Forget gate bias (reduced for exponential stability)
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        let w_q = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 0.1 })
            .init(device);
        let w_k = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 0.1 })
            .init(device);
        let w_v = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 0.1 })
            .init(device);

        let head_dim = hidden_size / num_heads;
        let ln = LayerNormConfig::new(head_dim).init(device);

        Self {
            weight_ih,
            weight_hh,
            bias: Param::from_tensor(bias),
            w_q,
            w_k,
            w_v,
            ln,
            input_size,
            hidden_size,
            num_heads,
        }
    }

    /// Forward pass through mLSTM cell consuming the state
    ///
    /// # Arguments
    /// * `input` - Input tensor [`batch_size`, `input_size`]
    /// * `state` - State to consume (moved)
    ///
    /// # Returns
    /// * New hidden state (for output)
    /// * New complete state
    pub fn forward_sequence(
        &self,
        input_seq: &Tensor<B, 3>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 3>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let [batch_size, seq_len, _] = input_seq.dims();
        let head_dim = self.hidden_size / self.num_heads;
        let device = input_seq.device();

        // 1. Parallel Projections (Q, K, V)
        let q = self.w_q.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, H, S, D_h]
        let k = self.w_k.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = self.w_v.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt();
        let scale_elem: <B as Backend>::FloatElem = num_traits::FromPrimitive::from_f64(scale).unwrap();
        let q = q / scale_elem;
        let k = k / scale_elem;

        // 2. Parallel Gates - CORRECCIÓN CRÍTICA AQUÍ
        let weight_ih_val = self.weight_ih.val().transpose();
        let bias_val = self.bias.val();

        // Proyección limpia sin unsqueeze encadenados
        let gates = input_seq.clone().matmul(weight_ih_val.reshape::<3, _>([1, self.input_size, 3 * self.hidden_size])) 
                    + bias_val.reshape::<3, _>([1, 1, 3 * self.hidden_size]);
        
        let chunks = gates.chunk(3, 2);
        
        let i_log = chunks[0].clone()
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2)
            .clamp(-6.0, 6.0); // Tighter clamp to prevent exp overflow
        let f_log = chunks[1].clone()
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2)
            .clamp(-6.0, 6.0); // Tighter clamp to prevent exp overflow
        let o = activation::sigmoid(chunks[2].clone()); // [B, S, D_hidden]

        let i_log_m = i_log.mean_dim(3); 
        let f_log_m = f_log.mean_dim(3); 

        // 3. Dual Form (Parallel Kernel)
        let indices = Tensor::<B, 1, burn::tensor::Int>::arange(0..seq_len as i64, &device);
        let row_idx = indices.clone().reshape::<2, _>([seq_len, 1]).expand::<2, _>([seq_len, seq_len]);
        let col_idx = indices.reshape::<2, _>([1, seq_len]).expand::<2, _>([seq_len, seq_len]);
        let mask_tri = row_idx.greater_equal(col_idx).float(); 
        
        let f_cumsum = mask_tri.reshape::<4, _>([1, 1, seq_len, seq_len]).matmul(f_log_m); 
        
        let log_weights = f_cumsum.clone() - f_cumsum.clone().swap_dims(2, 3) + i_log_m.clone().swap_dims(2, 3);
        
        let mask = self.get_causal_mask(seq_len, &device);
        let log_weights_masked = log_weights.mask_fill(mask, -1e10);

        // Global Max Stabilization (m_t)
        let m_0 = state.max_gate_log.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]); 
        let m_initial = f_cumsum.clone() + m_0.clone();
        
        let m_i_row = log_weights_masked.clone().max_dim(3); 
        let m_i = m_i_row.clone().mask_where(m_i_row.greater(m_initial.clone()), m_initial);
        
        // Evitar que m_i crezca sin control (Stability fix)
        let m_i_stable = m_i.clamp(-10.0, 10.0); 
        
        // CRÍTICO: Garantizar que log_diff <= 0 antes de exp (nunca explota)
        let log_diff = log_weights_masked - m_i_stable.clone();
        let weights = log_diff.clamp(-20.0, 0.0).exp(); // e^0=1, e^-20≈0, jamás >1 
        
        // Parallel Hidden State
        let q_k_t = q.clone().matmul(k.clone().swap_dims(2, 3));
        let h_parallel = (q_k_t * weights.clone()).matmul(v.clone()); 

        // Initial State Contribution con stabilization
        let initial_scale = (f_cumsum.clone() + m_0 - m_i_stable.clone()).exp().clamp(0.0, 1e10);
        let h_initial = q.matmul(state.cell.clone()) * initial_scale.clone();
        
        let h_heads = h_parallel + h_initial;

        // Normalizer
        let n_parallel = weights.clone().matmul(k.clone()); 
        let n_initial = state.normalizer.clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim])
            .expand::<4, _>([batch_size, self.num_heads, seq_len, head_dim]) * initial_scale.clone(); 
        let n_heads = (n_parallel + n_initial).clamp_min(1e-6);

        // 4. MHLN
        let h_normalized = h_heads / (n_heads.clone() + 1e-5);
        let h_reshaped = h_normalized.swap_dims(1, 2);
        let h_ln = self.ln.forward(h_reshaped); 
        
        let h_combined = h_ln.reshape::<3, _>([batch_size, seq_len, self.hidden_size]);
        let h_seq = (o * h_combined).clamp(-10.0, 10.0);

        // 5. Update State (Final T)
        let last_idx = seq_len - 1;
        let final_m = m_i_stable.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]).reshape::<3, _>([batch_size, self.num_heads, 1]);
        let final_norm = n_heads.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..head_dim]).reshape::<3, _>([batch_size, self.num_heads, head_dim]);
        
        let last_initial_scale = initial_scale.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]);
        let final_cell_initial = state.cell * last_initial_scale.reshape::<4, _>([batch_size, self.num_heads, 1, 1]);
        
        // ACTUALIZACIÓN DE MEMORIA PARALELA: sum_j w_j (v_j @ k_j^T)
        // Pesos de la última fila: [B, H, 1, S]
        let last_row_weights = weights.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..seq_len]);
        
        // Para calcular la suma de productos externos: (V .* w)^T @ K
        // v: [B, H, S, D], last_row_weights.T: [B, H, S, 1]
        let v_weighted = v * last_row_weights.swap_dims(2, 3); 
        let final_cell_update = v_weighted.swap_dims(2, 3).matmul(k); // [B, H, D, S] @ [B, H, S, D] -> [B, H, D, D]
        
        // Soft normalization (de la versión estable)
        let mut final_cell = final_cell_initial + final_cell_update;
        
        let c_abs_max_elem: <B as Backend>::FloatElem = final_cell.clone().abs().max().into_scalar();
        let c_abs_max_f64 = num_traits::ToPrimitive::to_f64(&c_abs_max_elem).unwrap_or(0.0);
        
        if c_abs_max_f64 > 1e-8 {
            let scale_factor_f64 = 1.0 / (1.0 + c_abs_max_f64 / 10.0);
            let scale_factor_elem: <B as Backend>::FloatElem =
                num_traits::FromPrimitive::from_f64(scale_factor_f64)
                    .expect("Failed to cast scale factor to backend float");
            
            final_cell = final_cell * scale_factor_elem;
        }

        let final_hidden = h_seq.clone()
            .slice([0..batch_size, last_idx..seq_len, 0..self.hidden_size])
            .reshape::<2, _>([batch_size, self.hidden_size]);

        (h_seq, MLstmstate::new(final_cell, final_hidden, final_norm, final_m))
    }

    fn get_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4, burn::tensor::Bool> {
        let indices = Tensor::<B, 1, burn::tensor::Int>::arange(0..seq_len as i64, device);
        let row_indices = indices.clone().reshape::<2, _>([seq_len, 1]).expand::<2, _>([seq_len, seq_len]);
        let col_indices = indices.reshape::<2, _>([1, seq_len]).expand::<2, _>([seq_len, seq_len]);
        col_indices.greater(row_indices).reshape::<4, _>([1, 1, seq_len, seq_len])
    }

    pub fn forward(
        &self,
        input: &Tensor<B, 2>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 2>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let MLstmstate { cell, hidden: _, normalizer, max_gate_log } = state;
        let [batch_size, _] = input.dims();
        let head_dim = self.hidden_size / self.num_heads;
        let _device = input.device();

        // Gates calculation (ignoring weight_hh for parallel consistency)
        let gates = input.clone().matmul(self.weight_ih.val().transpose())
            + self.bias.val().reshape::<2, _>([1, 3 * self.hidden_size]);

        let chunks = gates.chunk(3, 1);
        let i_log = chunks[0].clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim])
            .clamp(-6.0, 6.0);
        let f_log = chunks[1].clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim])
            .clamp(-6.0, 6.0);
        let o = activation::sigmoid(chunks[2].clone());

        // Projections
        let q = self.w_q.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let k = self.w_k.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let v = self.w_v.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, head_dim, 1]);

        let scale = (head_dim as f64).sqrt();
        let scale_elem: <B as Backend>::FloatElem = num_traits::FromPrimitive::from_f64(scale).unwrap();
        let q = q / scale_elem;
        let k = k / scale_elem;

        // Stabilization (Global)
        let i_log_m = i_log.mean_dim(3); // [B, H, 1, 1]
        let f_log_m = f_log.mean_dim(3); // [B, H, 1, 1]
        
        let m_0 = max_gate_log.reshape::<4, _>([batch_size, self.num_heads, 1, 1]); // [B, H, 1, 1]
        let f_plus_m_0 = f_log_m.clone() + m_0.clone();
        
        let m_t = i_log_m.clone().mask_where(i_log_m.clone().greater(f_plus_m_0.clone()), f_plus_m_0);
        let m_t_stable = m_t.clone().clamp(-10.0, 10.0);
        
        // Garantizar que los exponenciales no exploten
        let f_diff = (f_log_m + m_0 - m_t_stable.clone()).clamp(-20.0, 0.0);
        let i_diff = (i_log_m - m_t_stable.clone()).clamp(-20.0, 0.0);
        let f_stable = f_diff.exp();
        let i_stable = i_diff.exp();

        // Updates (Clamping added for symmetry with internal matrix update)
        let f_exp = f_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]); 
        let i_exp = i_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]);

        let cell_update = v.matmul(k.clone());
        let mut c_new = cell * f_exp.expand([batch_size, self.num_heads, head_dim, head_dim]) 
                   + cell_update * i_exp.expand([batch_size, self.num_heads, head_dim, head_dim]);
        
        // Soft normalization (de la versión estable)
        let c_abs_max_elem: <B as Backend>::FloatElem = c_new.clone().abs().max().into_scalar();
        let c_abs_max_f64 = num_traits::ToPrimitive::to_f64(&c_abs_max_elem).unwrap_or(0.0);
        
        if c_abs_max_f64 > 1e-8 {
            let scale_factor_f64 = 1.0 / (1.0 + c_abs_max_f64 / 10.0);
            let scale_factor_elem: <B as Backend>::FloatElem =
                num_traits::FromPrimitive::from_f64(scale_factor_f64)
                    .expect("Failed to cast scale factor to backend float");
            
            c_new = c_new * scale_factor_elem;
        }
        
        let n_new = normalizer * f_stable.reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]) 
                  + k.reshape::<3, _>([batch_size, self.num_heads, head_dim]) * i_stable.reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]);

        let h_heads = q.matmul(c_new.clone()).squeeze::<3>(2); 
        let n_heads = n_new.clone().clamp_min(1e-5);
        let h_normalized = h_heads / (n_heads + 1e-5);
        
        // LN per head
        let h_ln = self.ln.forward(h_normalized.reshape::<4, _>([batch_size, self.num_heads, 1, head_dim])).reshape::<2, _>([batch_size, self.hidden_size]);
        
        let h_combined = h_ln.reshape::<2, _>([batch_size, self.hidden_size]);
        let h_new = (o * h_combined).clamp(-10.0, 10.0);

        let new_state = MLstmstate::new(c_new, h_new.clone(), n_new, m_t.reshape::<3, _>([batch_size, self.num_heads, 1]));
        (h_new, new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_mlstm_forward() {
        let device = Default::default();
        let config = MLstmconfig::new(64, 128, 2).with_dropout(0.1);
        let mlstm = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = mlstm.forward(&input, None);

        assert_eq!(output.dims(), [4, 10, 128]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].hidden.dims(), [4, 128]);
        assert_eq!(states[0].cell.dims(), [4, 4, 32, 32]);
    }

    #[test]
    fn test_mlstm_cell() {
        let device = Default::default();
        let num_heads = 4;
        let cell = MLstmcell::new(32, 64, num_heads, &Initializer::XavierNormal { gain: 1.0 }, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let state = MLstmstate::new(
            Tensor::<TestBackend, 4>::zeros([4, num_heads, 16, 16], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 16], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 1], &device),
        );

        let (h_new, new_state) = cell.forward(&input, state);

        assert_eq!(h_new.dims(), [4, 64]);
        assert_eq!(new_state.cell.dims(), [4, 4, 16, 16]);
        assert_eq!(new_state.hidden.dims(), [4, 64]);
    }

    #[test]
    fn test_mlstm_state_reuse() {
        let device = Default::default();
        let config = MLstmconfig::new(32, 64, 1);
        let mlstm = config.init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);
        let input2 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);

        // First forward pass
        let (output1, states) = mlstm.forward(&input1, None);

        // Second forward pass reusing states
        let (output2, _final_states) = mlstm.forward(&input2, Some(states));

        assert_eq!(output1.dims(), [2, 5, 64]);
        assert_eq!(output2.dims(), [2, 5, 64]);
    }
}