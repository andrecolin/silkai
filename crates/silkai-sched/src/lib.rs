//! GPU capacity scheduler. No GPU types — only GB numbers.

#[cfg(feature = "test-util")]
pub mod clinic;
pub mod scheduler;
pub mod types;

pub use scheduler::{SchedError, Scheduler};
pub use types::*;
