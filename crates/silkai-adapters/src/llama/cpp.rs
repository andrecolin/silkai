use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::Inner;
use crate::{last_content, ChatMessage, EngineError, RunOptions};

const GPU_SHELF: u32 = 0;
const GPU_BENCH: u32 = 1000;
/// Prompt tokens are fed to the model in slices this long.
const PROMPT_CHUNK: usize = 512;

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
    messages: Vec<ChatMessage>,
    prefix: String,
    opts: RunOptions,
    cancel: CancellationToken,
) -> Result<mpsc::Receiver<String>, EngineError> {
    // Render and size the prompt now, so a request that cannot fit is
    // refused with a reason instead of an empty answer later.
    let tokens = {
        let g = inner.lock().expect("llama engine mutex");
        if !g.on_bench {
            return Err(EngineError::NotLoaded);
        }
        let model = g.model.as_ref().ok_or(EngineError::NotLoaded)?;
        prompt_tokens(model, &messages, &prefix, g.ctx_size as usize)?
    };
    let (tx, rx) = mpsc::channel(16);
    tokio::task::spawn_blocking(move || {
        if let Err(err) = generate(&inner, &tokens, &opts, tx.clone(), cancel) {
            eprintln!("llama.cpp generate failed: {err}");
        }
    });
    Ok(rx)
}

fn prompt_tokens(
    model: &LlamaModel,
    messages: &[ChatMessage],
    prefix: &str,
    n_ctx: usize,
) -> Result<Vec<LlamaToken>, EngineError> {
    let prompt = render_prompt(model, messages, prefix);
    // Templates emit their own BOS marker; adding another confuses most models.
    let tokens = model.str_to_token(&prompt, AddBos::Never).map_err(other)?;
    if tokens.is_empty() {
        return Err(EngineError::Rejected("empty prompt".into()));
    }
    if tokens.len() >= n_ctx {
        return Err(EngineError::Rejected(format!(
            "prompt is {} tokens but ctx_size is {n_ctx}; raise ctx_size on this model",
            tokens.len()
        )));
    }
    Ok(tokens)
}

/// Render the chat through the GGUF's own template (so instruct models see
/// their system/user/assistant markers), then append any already-streamed
/// prefix so a resumed run continues the same answer. A model without a
/// template gets the last message as plain text.
fn render_prompt(model: &LlamaModel, messages: &[ChatMessage], prefix: &str) -> String {
    let rendered =
        apply_template(model, messages).unwrap_or_else(|| last_content(messages).to_string());
    format!("{rendered}{prefix}")
}

fn apply_template(model: &LlamaModel, messages: &[ChatMessage]) -> Option<String> {
    let template = model.chat_template(None).ok()?;
    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<Result<_, _>>()
        .ok()?;
    model.apply_chat_template(&template, &chat, true).ok()
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
    tokens: &[LlamaToken],
    opts: &RunOptions,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    let g = inner.lock().expect("llama engine mutex");
    let model = g.model.as_ref().ok_or(EngineError::NotLoaded)?;
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(g.ctx_size))
        .with_n_batch(PROMPT_CHUNK as u32);
    let mut ctx = model.new_context(backend()?, params).map_err(other)?;
    let mut batch = LlamaBatch::new(PROMPT_CHUNK, 1);
    feed_prompt(&mut ctx, &mut batch, tokens)?;
    sample_loop(model, &mut ctx, batch, tokens.len(), opts, tx, cancel)
}

/// Decode the prompt in slices no longer than the batch, logits only on
/// the final token.
fn feed_prompt(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    tokens: &[LlamaToken],
) -> Result<(), EngineError> {
    let last = tokens.len() - 1;
    for (chunk_index, chunk) in tokens.chunks(PROMPT_CHUNK).enumerate() {
        batch.clear();
        for (j, token) in chunk.iter().enumerate() {
            let pos = chunk_index * PROMPT_CHUNK + j;
            batch
                .add(*token, pos as i32, &[0], pos == last)
                .map_err(other)?;
        }
        ctx.decode(batch).map_err(other)?;
    }
    Ok(())
}

fn sampler_for(opts: &RunOptions) -> LlamaSampler {
    match opts.temperature {
        Some(t) if t > 0.0 => {
            LlamaSampler::chain_simple([LlamaSampler::temp(t), LlamaSampler::dist(0)])
        }
        _ => LlamaSampler::greedy(),
    }
}

fn sample_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    mut batch: LlamaBatch<'_>,
    prompt_len: usize,
    opts: &RunOptions,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    let mut sampler = sampler_for(opts);
    let n_ctx = ctx.n_ctx() as usize;
    // New tokens are capped by the request, else by the room left in the window.
    let room = n_ctx - prompt_len;
    let limit = opts
        .max_tokens
        .map(|m| (m as usize).min(room))
        .unwrap_or(room);
    let mut pos = prompt_len;
    let mut produced = 0;
    while produced < limit && !cancel.is_cancelled() {
        if !emit_next(model, ctx, &mut batch, &mut sampler, pos, &tx)? {
            break;
        }
        pos += 1;
        produced += 1;
    }
    Ok(())
}

fn emit_next(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    sampler: &mut LlamaSampler,
    pos: usize,
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
    batch.add(token, pos as i32, &[0], true).map_err(other)?;
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
