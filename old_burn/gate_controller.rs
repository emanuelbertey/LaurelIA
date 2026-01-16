use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

/// Gate controller for LSTM-style gates.
/// Combines input and hidden transformations.
#[derive(Module, Debug)]
pub struct GateController<B: Backend> {
    /// Linear transformation for input
    pub input_transform: Linear<B>,
    /// Linear transformation for hidden state
    pub hidden_transform: Linear<B>,
}

impl<B: Backend> GateController<B> {
    /// Create a new gate controller
    pub fn new(d_input: usize, d_output: usize, bias: bool, device: &B::Device) -> Self {
        let input_transform = LinearConfig::new(d_input, d_output)
            .with_bias(bias)
            .init(device);
        let hidden_transform = LinearConfig::new(d_output, d_output)
            .with_bias(bias)
            .init(device);

        Self {
            input_transform,
            hidden_transform,
        }
    }

    /// Compute gate output: `input_transform(x)` + `hidden_transform(h)`
    pub fn forward(&self, input: Tensor<B, 2>, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.input_transform.forward(input) + self.hidden_transform.forward(hidden)
    }
}
