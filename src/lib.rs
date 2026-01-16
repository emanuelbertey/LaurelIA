/*
# xLSTM: Extended Long Short-Term Memory

This library implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM combines sLSTM (Scalar LSTM) and mLSTM (Matrix LSTM) in a novel
architecture to achieve state-of-the-art performance on various sequence
modeling tasks.

## Features

- **sLSTM**: Scalar LSTM with exponential gating
- **mLSTM**: Matrix LSTM with matrix memory state
- **xLSTMBlock**: Flexible block combining sLSTM or mLSTM with normalization
- **xLSTM**: Main model supporting mixed block architectures
- **Per-block learning rates**: Support for different learning rates for different block types

## Example

```rust,no_run
use burn::backend::NdArray;
use xlstm::{xLSTM, xLSTMConfig, LearningRateConfig};

type Backend = NdArray;

let device = Default::default();
let config = xLSTMConfig::new(128, 256, 2, 4, 128)
    .with_dropout(0.1)
    .with_lstm_type(xlstm::LstmType::Alternate);

let model = config.init::<Backend>(&device);

// Configure per-block learning rates
let lr_config = LearningRateConfig::per_block_type(
    1e-4,  // sLSTM learning rate
    1e-5,  // mLSTM learning rate
    1e-4,  // Other components learning rate
);

// During training:
// model = model.optimizer_step(&lr_config, &mut optimizer, gradients);
```
*/

extern crate alloc;

mod block;
mod gate_controller;
mod mlstm;
mod model;
mod slstm;

pub use block::{BlockType, XLstmblock, XLstmblockConfig, LSTMState};
pub use gate_controller::GateController;
pub use mlstm::{MLstm, MLstmcell, MLstmconfig, MLstmstate};
pub use model::{LstmType, XLstm, XLstmconfig};
pub use slstm::{SLstm, SLstmcell, SLstmconfig, SLstmstate};

pub const VERSION: &str = "0.1.0";