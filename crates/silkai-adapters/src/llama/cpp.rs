use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::Inner;
use crate::EngineError;

const GPU_SHELF: u32 = 0;
const GPU_BENCH: u32 = 1000;
const MAX_NEW_TOKENS: i32 = 256;

pub async fn place(
    inner: Arc<Mutex<Inner>>,
    path: String,
    bench: bool,
    gpu: u32,
) -> Result<(), EngineError> {
    let layers = if bench { GPU_BENCH } else { GPU_SHELF };
    tokio::task::spawn_blocking(move || place_sync(&inner, &path, layers, bench, gpu))
        .await
        .map_err(join_err)?
}

pub fn start_run(
    inner: Arc<Mutex<Inner>>,
    prompt: String,
    prefix: String,
    cancel: CancellationToken,
) -> Result<mpsc::Receiver<String>, EngineError> {
    if !inner.lock().expect("llama engine mutex").on_bench {
        return Err(EngineError::NotLoaded);
    }
    let (tx, rx) = mpsc::channel(16);
    tokio::task::spawn_blocking(move || {
        let input = if prefix.is_empty() {
            prompt
        } else {
            format!("{prompt}{prefix}")
        };
        let _ = generate(&inner, &input, tx, cancel);
    });
    Ok(rx)
}

fn place_sync(
    inner: &Mutex<Inner>,
    path: &str,
    layers: u32,
    bench: bool,
    gpu: u32,
) -> Result<(), EngineError> {
    let model = load_model(path, layers, gpu)?;
    let mut g = inner.lock().expect("llama engine mutex");
    g.model = Some(model);
    g.path = Some(path.to_string());
    g.on_bench = bench;
    Ok(())
}

fn load_model(path: &str, layers: u32, gpu: u32) -> Result<LlamaModel, EngineError> {
    if !std::path::Path::new(path).exists() {
        return Err(EngineError::Other(format!("missing file: {path}")));
    }
    let backend = backend()?;
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(layers)
        .with_main_gpu(i32::try_from(gpu).unwrap_or(0));
    LlamaModel::load_from_file(backend, path, &params).map_err(other)
}

fn generate(
    inner: &Mutex<Inner>,
    prompt: &str,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    let g = inner.lock().expect("llama engine mutex");
    let model = g.model.as_ref().ok_or(EngineError::NotLoaded)?;
    let mut ctx = model
        .new_context(backend()?, LlamaContextParams::default())
        .map_err(other)?;
    infer(model, &mut ctx, prompt, tx, cancel)
}

fn infer(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    prompt: &str,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    let tokens = model.str_to_token(prompt, AddBos::Always).map_err(other)?;
    if tokens.is_empty() {
        return Err(EngineError::Other("empty prompt tokens".into()));
    }
    let mut batch = LlamaBatch::new(tokens.len() + 64, 1);
    fill_prompt(&mut batch, &tokens)?;
    ctx.decode(&mut batch).map_err(other)?;
    sample_loop(model, ctx, batch, tx, cancel)
}

fn fill_prompt(batch: &mut LlamaBatch<'_>, tokens: &[LlamaToken]) -> Result<(), EngineError> {
    let last = tokens.len() - 1;
    for (i, token) in tokens.iter().enumerate() {
        batch
            .add(*token, i as i32, &[0], i == last)
            .map_err(other)?;
    }
    Ok(())
}

fn sample_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    mut batch: LlamaBatch<'_>,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    let mut sampler = LlamaSampler::greedy();
    let mut n_cur = batch.n_tokens();
    let n_ctx = ctx.n_ctx() as i32;
    while n_cur < n_ctx && n_cur < MAX_NEW_TOKENS && !cancel.is_cancelled() {
        if !emit_next(model, ctx, &mut batch, &mut sampler, n_cur, &tx)? {
            break;
        }
        n_cur += 1;
    }
    Ok(())
}

fn emit_next(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    sampler: &mut LlamaSampler,
    n_cur: i32,
    tx: &mpsc::Sender<String>,
) -> Result<bool, EngineError> {
    let token = sampler.sample(ctx, batch.n_tokens() - 1);
    sampler.accept(token);
    if model.is_eog_token(token) {
        return Ok(false);
    }
    if !send_piece(model, token, tx)? {
        return Ok(false);
    }
    batch.clear();
    batch.add(token, n_cur, &[0], true).map_err(other)?;
    ctx.decode(batch).map_err(other)?;
    Ok(true)
}

fn send_piece(
    model: &LlamaModel,
    token: LlamaToken,
    tx: &mpsc::Sender<String>,
) -> Result<bool, EngineError> {
    #[allow(deprecated)]
    let piece = model
        .token_to_str(token, llama_cpp_2::model::Special::Plaintext)
        .map_err(other)?;
    Ok(tx.blocking_send(piece).is_ok())
}

fn backend() -> Result<&'static LlamaBackend, EngineError> {
    static CELL: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    match CELL.get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string())) {
        Ok(backend) => Ok(backend),
        Err(err) => Err(EngineError::Other(err.clone())),
    }
}

fn other(err: impl ToString) -> EngineError {
    EngineError::Other(err.to_string())
}

fn join_err(err: tokio::task::JoinError) -> EngineError {
    EngineError::Other(err.to_string())
}
