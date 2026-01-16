/*!
# sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer},
    tensor::{Tensor, activation, backend::Backend},
};

/// State for sLSTM containing cell and hidden states
#[derive(Clone, Debug)]
pub struct SLstmstate<B: Backend, const D: usize> {
    /// Cell state
    pub cell: Tensor<B, D>,
    /// Normalizer state (for stable exponential gating)
    pub normalizer: Tensor<B, D>,
    /// Hidden state
    pub hidden: Tensor<B, D>,
    /// Stabilizer state (log-space max tracker)
    pub stabilizer: Tensor<B, D>,
}

impl<B: Backend, const D: usize> SLstmstate<B, D> {
    /// Create a new sLSTM state
    pub const fn new(
        cell: Tensor<B, D>,
        normalizer: Tensor<B, D>,
        hidden: Tensor<B, D>,
        stabilizer: Tensor<B, D>,
    ) -> Self {
        Self {
            cell,
            normalizer,
            hidden,
            stabilizer,
        }
    }
}

/// Configuration for sLSTM
#[derive(Config, Debug)]
pub struct SLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}

impl SLstmconfig {
    /// Initialize a new sLSTM
    pub fn init<B: Backend>(&self, device: &B::Device) -> SLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                SLstmcell::new(input_size, self.d_hidden, &self.initializer, device)
            })
            .collect();

        SLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
        }
    }
}

/// sLSTM layer implementation
#[derive(Module, Debug)]
pub struct SLstm<B: Backend> {
    /// Stack of sLSTM cells
    pub layers: alloc::vec::Vec<SLstmcell<B>>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl<B: Backend> SLstm<B> {
    /// Forward pass through sLSTM consuming and returning states
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
        states: Option<alloc::vec::Vec<SLstmstate<B, 2>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<SLstmstate<B, 2>>) {
        let device = input_seq.device();
        let [batch_size, seq_length, _] = input_seq.dims();

        // Initialize or consume provided states
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));

        let mut all_outputs = alloc::vec::Vec::with_capacity(seq_length);

        for t in 0..seq_length {
            let input_t = input_seq
                .clone()
                .slice([0..batch_size, t..(t + 1), 0..self.d_input])
                .squeeze(1);

            let mut layer_input = input_t;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // Take ownership of the state using mem::take (replaces with default/zeros)
                let old_state = core::mem::replace(
                    &mut hidden_states[layer_idx],
                    SLstmstate::new(
                        Tensor::zeros([batch_size, self.d_hidden], &device),
                        Tensor::zeros([batch_size, self.d_hidden], &device),
                        Tensor::zeros([batch_size, self.d_hidden], &device),
                        Tensor::zeros([batch_size, self.d_hidden], &device),
                    ),
                );

                // Consume the state and get new state back
                let (h_new, new_state) = layer.forward(layer_input, old_state);

                // Store the new state
                hidden_states[layer_idx] = new_state;

                // Apply dropout between layers (but not after last layer)
                layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                    self.dropout_layer.forward(h_new)
                } else {
                    h_new
                };
            }

            all_outputs.push(layer_input.unsqueeze_dim(1));
        }

        let output = Tensor::cat(all_outputs, 1);
        (output, hidden_states)
    }

    /// Initialize hidden states
    fn init_hidden(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> alloc::vec::Vec<SLstmstate<B, 2>> {
        (0..self.num_layers)
            .map(|_| {
                SLstmstate::new(
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                )
            })
            .collect()
    }
}

/// sLSTM cell implementation with exponential gating
#[derive(Module, Debug)]
pub struct SLstmcell<B: Backend> {
    /// Weight matrix for input to gates
    pub weight_ih: Param<Tensor<B, 2>>,
    /// Weight matrix for hidden to gates
    pub weight_hh: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl<B: Backend> SLstmcell<B> {
    /// Create a new sLSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        initializer: &Initializer,
        device: &B::Device,
    ) -> Self {
        // 4 gates: input, forget, cell, output
        // For weight matrix [output_size, input_size]: fan_in=input_size, fan_out=output_size
        let weight_ih = initializer.init_with(
            [4 * hidden_size, input_size],
            Some(input_size),
            Some(4 * hidden_size),
            device,
        );
        let weight_hh = initializer.init_with(
            [4 * hidden_size, hidden_size],
            Some(hidden_size),
            Some(4 * hidden_size),
            device,
        );

        // Initialize biases with specific values for stability
        let mut bias_data = alloc::vec![0.0; 4 * hidden_size];
        for item in bias_data.iter_mut().take(hidden_size) {
            *item = -2.0;
        }
        for item in bias_data.iter_mut().take(2 * hidden_size).skip(hidden_size) {
            *item = -2.0;
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        Self {
            weight_ih,
            weight_hh,
            bias: Param::from_tensor(bias),
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through sLSTM cell consuming the state
    ///
    /// # Arguments
    /// * `input` - Input tensor [`batch_size`, `input_size`]
    /// * `state` - State to consume (moved)
    ///
    /// # Returns
    /// * New hidden state (for output)
    /// * New complete state
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        state: SLstmstate<B, 2>,
    ) -> (Tensor<B, 2>, SLstmstate<B, 2>) {
        let SLstmstate {
            cell,
            normalizer,
            hidden,
            stabilizer,
        } = state;

        // Compute all gates: i, f, g, o
        let gates = input.matmul(self.weight_ih.val().transpose())
            + hidden.matmul(self.weight_hh.val().transpose())
            + self.bias.val().unsqueeze_dim(0);

        let chunks = gates.chunk(4, 1);
        let i_gate = chunks[0].clone();
        let f_gate = chunks[1].clone();
        let g_gate = chunks[2].clone();
        let o_gate = chunks[3].clone();

        let m_prev_plus_f = stabilizer.clone() + f_gate.clone();
        let m_new = Tensor::max_pair(m_prev_plus_f.clone(), i_gate.clone());

        let i_exp = (i_gate - m_new.clone()).clamp(-20.0, 0.0).exp();
        let f_exp = (m_prev_plus_f - m_new.clone()).clamp(-20.0, 0.0).exp();

        let g = g_gate.tanh();
        let o = activation::sigmoid(o_gate);

        let c_new = f_exp.clone() * cell + i_exp.clone() * g;
        let n_new = f_exp * normalizer + i_exp;

        let n_safe = n_new.clone().clamp_min(1e-6);
        let h_new = o * (c_new.clone() / n_safe).tanh();

        let new_state = SLstmstate::new(c_new, n_new, h_new.clone(), m_new);
        (h_new, new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_slstm_forward() {
        let device = Default::default();
        let config = SLstmconfig::new(64, 128, 2).with_dropout(0.1);
        let slstm = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = slstm.forward(&input, None);

        assert_eq!(output.dims(), [4, 10, 128]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].hidden.dims(), [4, 128]);
        assert_eq!(states[0].cell.dims(), [4, 128]);
        assert_eq!(states[0].normalizer.dims(), [4, 128]);
        assert_eq!(states[0].stabilizer.dims(), [4, 128]);
    }

    #[test]
    fn test_slstm_cell() {
        let device = Default::default();
        let cell = SLstmcell::new(32, 64, &Initializer::XavierNormal { gain: 1.0 }, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let state = SLstmstate::new(
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
        );

        let (h_new, new_state) = cell.forward(input, state);

        assert_eq!(h_new.dims(), [4, 64]);
        assert_eq!(new_state.cell.dims(), [4, 64]);
        assert_eq!(new_state.hidden.dims(), [4, 64]);
        assert_eq!(new_state.normalizer.dims(), [4, 64]);
        assert_eq!(new_state.stabilizer.dims(), [4, 64]);
    }

    #[test]
    fn test_slstm_state_reuse() {
        let device = Default::default();
        let config = SLstmconfig::new(32, 64, 1);
        let slstm = config.init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);
        let input2 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);

        // First forward pass
        let (output1, states) = slstm.forward(&input1, None);

        // Second forward pass reusing states
        let (output2, _final_states) = slstm.forward(&input2, Some(states));

        assert_eq!(output1.dims(), [2, 5, 64]);
        assert_eq!(output2.dims(), [2, 5, 64]);
    }
}
