use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder, Linear, linear};

/// Gate controller for LSTM-style gates.
/// Combines input and hidden transformations.
#[derive(Debug)]
pub struct GateController {
    /// Linear transformation for input
    pub input_transform: Linear,
    /// Linear transformation for hidden state
    pub hidden_transform: Linear,
}

impl GateController {
    /// Create a new gate controller
    pub fn new(d_input: usize, d_output: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let input_transform = if bias {
            linear(d_input, d_output, vb.pp("input_transform"))?
        } else {
            candle_nn::linear_no_bias(d_input, d_output, vb.pp("input_transform"))?
        };
        
        let hidden_transform = if bias {
            linear(d_output, d_output, vb.pp("hidden_transform"))?
        } else {
            candle_nn::linear_no_bias(d_output, d_output, vb.pp("hidden_transform"))?
        };

        Ok(Self {
            input_transform,
            hidden_transform,
        })
    }

    /// Compute gate output: `input_transform(x)` + `hidden_transform(h)`
    pub fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        Ok((self.input_transform.forward(input)? + self.hidden_transform.forward(hidden)?)?)
    }
}
