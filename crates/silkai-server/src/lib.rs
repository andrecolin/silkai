//! The SilkAI daemon: config loading, the runtime that applies scheduler
//! decisions to engines, and the HTTP/WebSocket surface (OpenAI-style chat,
//! status, events, metrics, an optional status page).
//!
//! Most users want the `silkai` binary. Embed this crate when you need the
//! daemon inside another process: build an [`config::AppConfig`], then
//! [`serve`] it.

pub mod app;
pub(crate) mod auth;
pub mod config;
pub mod events;
pub mod metrics;
pub mod runtime;
pub mod sampler;
pub mod status;
pub(crate) mod ws;

mod serve;

pub use runtime::Runtime;
pub use serve::{serve, serve_listener};
