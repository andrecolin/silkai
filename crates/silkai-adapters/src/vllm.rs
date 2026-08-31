use std::sync::{Mutex, MutexGuard};

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{Engine, EngineError};

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
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError> {
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
        let prompt = prompt.to_string();
        tokio::spawn(async move {
            stream_chat(client, url, model, prompt, tx, cancel).await;
        });
        Ok(rx)
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }
}

async fn stream_chat(
    client: reqwest::Client,
    url: String,
    model: String,
    prompt: String,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
    });
    let send = client.post(url).json(&body).send();
    let resp = tokio::select! {
        _ = cancel.cancelled() => return,
        result = send => match result {
            Ok(resp) if resp.status().is_success() => resp,
            _ => return,
        }
    };
    let mut resp = resp;
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            chunk = resp.chunk() => match chunk {
                Ok(Some(bytes)) => {
                    if !emit_sse(&mut buf, &bytes, &tx).await {
                        return;
                    }
                }
                _ => return,
            }
        }
    }
}

async fn emit_sse(buf: &mut String, bytes: &[u8], tx: &mpsc::Sender<String>) -> bool {
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
        if let Some(text) = delta_content(data) {
            if tx.send(text).await.is_err() {
                return false;
            }
        }
    }
    true
}

fn delta_content(data: &str) -> Option<String> {
    let chunk: StreamChunk = serde_json::from_str(data).ok()?;
    let text = chunk.choices.into_iter().next()?.delta.content?;
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn http_err(err: reqwest::Error) -> EngineError {
    EngineError::Other(err.to_string())
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}
