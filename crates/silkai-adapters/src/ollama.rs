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

pub struct OllamaEngine {
    vram_gb: f64,
    url: String,
    client: reqwest::Client,
    inner: Mutex<Inner>,
}

impl OllamaEngine {
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
        self.inner.lock().expect("ollama engine mutex")
    }

    fn on_bench(&self) -> bool {
        self.lock().on_bench
    }

    fn stored_model(&self) -> Result<String, EngineError> {
        self.lock().model.clone().ok_or(EngineError::NotLoaded)
    }

    async fn generate(&self, model: &str, keep_alive: i64) -> Result<(), EngineError> {
        let url = format!("{}/api/generate", self.url);
        let body = serde_json::json!({
            "model": model,
            "keep_alive": keep_alive,
        });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(http_err)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(EngineError::Other(format!(
                "ollama {url} {}",
                resp.status()
            )))
        }
    }
}

#[async_trait]
impl Engine for OllamaEngine {
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
        self.generate(path, -1).await?;
        self.lock().on_bench = true;
        Ok(())
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        let model = self.stored_model()?;
        self.lock().gpu = Some(gpu);
        self.generate(&model, -1).await?;
        self.lock().on_bench = true;
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        let model = self.stored_model()?;
        self.generate(&model, 0).await?;
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
        prefix: &str,
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
        let url = format!("{}/api/chat", self.url);
        let prompt = prompt.to_string();
        let prefix = prefix.to_string();
        tokio::spawn(async move {
            stream_chat(client, url, model, prompt, prefix, tx, cancel).await;
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
    prefix: String,
    tx: mpsc::Sender<String>,
    cancel: CancellationToken,
) {
    let mut messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    if !prefix.is_empty() {
        messages.push(serde_json::json!({"role": "assistant", "content": prefix}));
    }
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "keep_alive": -1,
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
                    if !emit_ndjson(&mut buf, &bytes, &tx).await {
                        return;
                    }
                }
                _ => return,
            }
        }
    }
}

async fn emit_ndjson(buf: &mut String, bytes: &[u8], tx: &mpsc::Sender<String>) -> bool {
    buf.push_str(&String::from_utf8_lossy(bytes));
    while let Some(i) = buf.find('\n') {
        let mut line: String = buf.drain(..=i).collect();
        if line.ends_with('\n') {
            line.pop();
        }
        if line.ends_with('\r') {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        let Some(chunk) = parse_line(&line) else {
            continue;
        };
        let done = chunk.done;
        if let Some(text) = chunk.content() {
            if !text.is_empty() && tx.send(text).await.is_err() {
                return false;
            }
        }
        if done {
            return false;
        }
    }
    true
}

fn parse_line(line: &str) -> Option<ChatLine> {
    serde_json::from_str(line).ok()
}

fn http_err(err: reqwest::Error) -> EngineError {
    EngineError::Other(err.to_string())
}

#[derive(Deserialize)]
struct ChatLine {
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    done: bool,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

impl ChatLine {
    fn content(self) -> Option<String> {
        self.message?.content
    }
}
