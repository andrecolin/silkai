use std::sync::{Mutex, MutexGuard};

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{ChatMessage, Chunk, Engine, EngineError, RunEnd, RunOptions, Usage};

struct Inner {
    on_bench: bool,
    gpu: Option<u32>,
    model: Option<String>,
}

pub struct VllmEngine {
    vram_gb: f64,
    url: String,
    client: reqwest::Client,
    inner: Mutex<Inner>,
}

impl VllmEngine {
    pub fn new(_name: &str, vram_gb: f64, url: impl AsRef<str>) -> Self {
        Self {
            vram_gb,
            url: url.as_ref().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
            inner: Mutex::new(Inner {
                on_bench: false,
                gpu: None,
                model: None,
            }),
        }
    }

    pub fn gpu(&self) -> Option<u32> {
        self.lock().gpu
    }

    /// Record that the model is on `gpu` without talking to the server.
    /// The process engine uses this after it has spawned and health-checked
    /// a child that starts awake.
    pub(crate) fn mark_on_bench(&self, gpu: u32) {
        let mut inner = self.lock();
        inner.gpu = Some(gpu);
        inner.on_bench = true;
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().expect("vllm engine mutex")
    }

    fn on_bench(&self) -> bool {
        self.lock().on_bench
    }

    fn stored_model(&self) -> Result<String, EngineError> {
        self.lock().model.clone().ok_or(EngineError::NotLoaded)
    }

    async fn post(&self, path: &str) -> Result<(), EngineError> {
        let url = format!("{}{path}", self.url);
        let resp = self.client.post(&url).send().await.map_err(http_err)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(EngineError::Other(format!("vllm {url} {}", resp.status())))
        }
    }
}

#[async_trait]
impl Engine for VllmEngine {
    async fn warm(&self, path: &str) -> Result<(), EngineError> {
        self.lock().model = Some(path.to_string());
        Ok(())
    }

    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError> {
        {
            let mut inner = self.lock();
            inner.model = Some(path.to_string());
            inner.gpu = Some(gpu);
        }
        self.post("/wake_up").await?;
        self.lock().on_bench = true;
        Ok(())
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        self.lock().gpu = Some(gpu);
        self.post("/wake_up").await?;
        self.lock().on_bench = true;
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.post("/sleep?level=1").await?;
        self.lock().on_bench = false;
        Ok(())
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.lock().on_bench = false;
        Ok(())
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<Chunk>, EngineError> {
        if !self.on_bench() {
            return Err(EngineError::NotLoaded);
        }
        let model = self.stored_model()?;
        let (tx, rx) = mpsc::channel(16);
        if cancel.is_cancelled() {
            return Ok(rx);
        }
        let client = self.client.clone();
        let url = format!("{}/v1/chat/completions", self.url);
        let messages = with_prefix(messages, prefix);
        let opts = opts.clone();
        tokio::spawn(async move {
            stream_chat(client, url, model, messages, opts, tx, cancel).await;
        });
        Ok(rx)
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }
}

/// The request messages, plus the already-streamed `prefix` as a trailing
/// assistant turn so the server continues instead of starting over.
pub(crate) fn with_prefix(messages: &[ChatMessage], prefix: &str) -> Vec<ChatMessage> {
    let mut out = messages.to_vec();
    if !prefix.is_empty() {
        out.push(ChatMessage::assistant(prefix));
    }
    out
}

async fn stream_chat(
    client: reqwest::Client,
    url: String,
    model: String,
    messages: Vec<ChatMessage>,
    opts: RunOptions,
    tx: mpsc::Sender<Chunk>,
    cancel: CancellationToken,
) {
    // `stream_options` is what makes an OpenAI-shaped server put a usage
    // record in the stream; without it the counts never arrive. The finish
    // reason rides on the last content chunk either way.
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(m) = opts.max_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }
    if let Some(t) = opts.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    let send = client.post(url).json(&body).send();
    let resp = tokio::select! {
        _ = cancel.cancelled() => return,
        result = send => match result {
            Ok(resp) if resp.status().is_success() => resp,
            Ok(resp) => {
                let _ = tx.send(Chunk::Reject(rejection_of(resp).await)).await;
                return;
            }
            Err(err) => {
                let _ = tx.send(Chunk::Reject(format!("engine request failed: {err}"))).await;
                return;
            }
        }
    };
    let mut resp = resp;
    let mut buf = String::new();
    let mut end = RunEnd::default();
    loop {
        let more = tokio::select! {
            // A cancelled run is preempted, not finished: it has no end to
            // report, and the job resumes from the tokens already sent.
            _ = cancel.cancelled() => return,
            chunk = resp.chunk() => match chunk {
                Ok(Some(bytes)) => emit_sse(&mut buf, &bytes, &tx, &mut end).await,
                _ => false,
            }
        };
        if !more {
            break;
        }
    }
    if end != RunEnd::default() {
        let _ = tx.send(Chunk::End(end)).await;
    }
}

/// Reads SSE lines into tokens, keeping whatever the stream says about how
/// the run ended. Returns false once the stream is finished or the receiver
/// is gone; `end` holds what was seen so the caller can send it last.
async fn emit_sse(
    buf: &mut String,
    bytes: &[u8],
    tx: &mpsc::Sender<Chunk>,
    end: &mut RunEnd,
) -> bool {
    buf.push_str(&String::from_utf8_lossy(bytes));
    while let Some(i) = buf.find('\n') {
        let mut line: String = buf.drain(..=i).collect();
        if line.ends_with('\n') {
            line.pop();
        }
        if line.ends_with('\r') {
            line.pop();
        }
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return false;
        }
        // Some engines answer a rejection with an in-stream error object
        // rather than a non-2xx status. Surface it the same way.
        if let Ok(err) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(msg) = err
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .filter(|m| !m.is_empty())
            {
                let _ = tx.send(Chunk::Reject(msg.to_string())).await;
                return false;
            }
        }
        let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) else {
            continue;
        };
        // The usage record arrives in its own trailing chunk, which carries
        // an empty `choices`, so it is read before the choices are touched.
        if let Some(usage) = chunk.usage {
            end.usage = Some(usage);
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            continue;
        };
        if let Some(reason) = choice.finish_reason {
            end.finish_reason = Some(reason);
        }
        // A reasoning engine sends its trace as its own deltas before any
        // content arrives. Forwarding them is what lets a client show the
        // model thinking instead of sitting silent; dropping them here is
        // indistinguishable from a stalled stream.
        if let Some(text) = choice.delta.reasoning_content.filter(|t| !t.is_empty()) {
            if tx.send(Chunk::Reasoning(text)).await.is_err() {
                return false;
            }
        }
        if let Some(text) = choice.delta.content.filter(|t| !t.is_empty()) {
            if tx.send(Chunk::Token(text)).await.is_err() {
                return false;
            }
        }
    }
    true
}

fn http_err(err: reqwest::Error) -> EngineError {
    EngineError::Other(err.to_string())
}

/// Turn a non-2xx engine response into the reason a client should see.
/// OpenAI-shaped errors carry `error.message`; anything else is truncated
/// raw text, with the status as the last resort.
async fn rejection_of(resp: reqwest::Response) -> String {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let body = body.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(msg) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty())
        {
            return msg.to_string();
        }
    }
    if !body.is_empty() {
        let mut s = body.to_string();
        s.truncate(500);
        return s;
    }
    format!("engine returned {status}")
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    /// llama.cpp under `--reasoning-format deepseek`, and other engines that
    /// separate the trace from the answer, put it here.
    #[serde(default)]
    reasoning_content: Option<String>,
}
