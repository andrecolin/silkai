//! A managed stable-diffusion.cpp `sd-server` that answers a chat turn with
//! a generated video. The last user message is the prompt (an attached
//! image, if any, is the first frame); the reply is one assistant message
//! carrying the finished clip as a `<video>` tag and a link, and the clip
//! itself is written to a directory the server hands out under
//! `/v1/files/{model}/`.
//!
//! sd-server has no `/health`; `GET /sdcpp/v1/capabilities` is what answers
//! once the weights are loaded. Generation is a queued job: submit, poll,
//! collect. Progress goes out as reasoning chunks so a client shows the
//! wait instead of a stalled stream.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::child::ManagedChild;
use crate::{ChatMessage, Chunk, Content, Engine, EngineError, RunEnd, RunOptions};

const POLL: Duration = Duration::from_secs(1);
/// How often a run still generating says so, so the client's stream is
/// never silent for long.
const HEARTBEAT: Duration = Duration::from_secs(15);

/// Where finished clips go, and how a reply refers to them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdcppOutput {
    /// Directory the clip is written to. Created on first use.
    pub dir: PathBuf,
    /// Prefix for the link in the reply, before `/v1/files/...`. Empty
    /// gives a link relative to whatever host served the chat; set it to
    /// the URL the client reaches SilkAI at when that is behind a gateway.
    pub link_base: String,
}

struct Inner {
    on_bench: bool,
    gpu: Option<u32>,
}

pub struct SdcppEngine {
    name: String,
    vram_gb: f64,
    url: String,
    child: ManagedChild,
    client: reqwest::Client,
    output: SdcppOutput,
    /// Extra `vid_gen` request fields from the config (`width`,
    /// `video_frames`, `sample_params`, ...), sent under the prompt.
    params: serde_json::Value,
    inner: Mutex<Inner>,
}

impl SdcppEngine {
    pub fn new(
        name: &str,
        vram_gb: f64,
        url: impl AsRef<str>,
        cmd: Vec<String>,
        output: SdcppOutput,
        params: serde_json::Value,
    ) -> Self {
        Self {
            name: name.to_string(),
            vram_gb,
            url: url.as_ref().trim_end_matches('/').to_string(),
            child: ManagedChild::new(cmd),
            client: reqwest::Client::new(),
            output,
            params,
            inner: Mutex::new(Inner {
                on_bench: false,
                gpu: None,
            }),
        }
    }

    pub fn alive(&self) -> bool {
        self.child.alive()
    }

    pub fn gpu(&self) -> Option<u32> {
        self.inner.lock().expect("sdcpp engine mutex").gpu
    }

    fn on_bench(&self) -> bool {
        self.inner.lock().expect("sdcpp engine mutex").on_bench
    }

    fn set_bench(&self, gpu: Option<u32>) {
        let mut inner = self.inner.lock().expect("sdcpp engine mutex");
        inner.on_bench = gpu.is_some();
        inner.gpu = gpu;
    }

    async fn park(&self) -> Result<(), EngineError> {
        self.child.kill().await?;
        self.set_bench(None);
        Ok(())
    }
}

#[async_trait]
impl Engine for SdcppEngine {
    /// sd-server loads whatever its command line names; `path` is only the
    /// roster's label for it.
    async fn warm(&self, _path: &str) -> Result<(), EngineError> {
        Ok(())
    }

    async fn load(&self, _path: &str, gpu: u32) -> Result<(), EngineError> {
        self.wake(gpu).await
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        let ready = format!("{}/sdcpp/v1/capabilities", self.url);
        self.child.spawn_ready(gpu, &ready).await?;
        self.set_bench(Some(gpu));
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.park().await
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.park().await
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        _opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<Chunk>, EngineError> {
        if !self.on_bench() {
            return Err(EngineError::NotLoaded);
        }
        let (tx, rx) = mpsc::channel(16);
        if cancel.is_cancelled() {
            return Ok(rx);
        }
        // The reply is one message sent whole, so a resumed run that already
        // streamed it has nothing left but its end.
        if !prefix.is_empty() {
            let _ = tx.send(Chunk::End(RunEnd::reason("stop"))).await;
            return Ok(rx);
        }
        let Some(request) = Request::from_messages(messages) else {
            let _ = tx
                .send(Chunk::Reject("no user prompt to generate from".into()))
                .await;
            return Ok(rx);
        };
        let job = Job {
            client: self.client.clone(),
            url: self.url.clone(),
            body: request.body(&self.params),
            model: self.name.clone(),
            output: self.output.clone(),
        };
        tokio::spawn(async move {
            job.generate(tx, cancel).await;
        });
        Ok(rx)
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }

    fn pid(&self) -> Option<u32> {
        self.child.id()
    }
}

/// What the last user turn asked for.
struct Request {
    prompt: String,
    /// The first image attached to that turn, as the client sent it: a data
    /// URL or raw base64, both of which sd-server accepts.
    init_image: Option<String>,
}

impl Request {
    fn from_messages(messages: &[ChatMessage]) -> Option<Self> {
        let turn = messages.iter().rev().find(|m| m.role == "user")?;
        let prompt = turn.content.text().trim().to_string();
        if prompt.is_empty() {
            return None;
        }
        Some(Self {
            prompt,
            init_image: first_image(&turn.content),
        })
    }

    fn body(&self, params: &serde_json::Value) -> serde_json::Value {
        let mut body = match params {
            serde_json::Value::Object(_) => params.clone(),
            _ => serde_json::json!({}),
        };
        body["prompt"] = serde_json::Value::String(self.prompt.clone());
        if let Some(image) = &self.init_image {
            body["init_image"] = serde_json::Value::String(image.clone());
        }
        body
    }
}

fn first_image(content: &Content) -> Option<String> {
    let Content::Parts(parts) = content else {
        return None;
    };
    parts.iter().find_map(|p| {
        let url = p.get("image_url")?;
        let url = url.get("url").unwrap_or(url).as_str()?;
        (!url.is_empty()).then(|| url.to_string())
    })
}

struct Job {
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
    model: String,
    output: SdcppOutput,
}

impl Job {
    async fn generate(self, tx: mpsc::Sender<Chunk>, cancel: CancellationToken) {
        let id = tokio::select! {
            _ = cancel.cancelled() => return,
            submitted = self.submit() => match submitted {
                Ok(id) => id,
                Err(reason) => {
                    let _ = tx.send(Chunk::Reject(reason)).await;
                    return;
                }
            }
        };
        let outcome = tokio::select! {
            // A cancelled run is preempted or abandoned; either way the
            // server should stop spending the card on it.
            _ = cancel.cancelled() => {
                let _ = self.client.post(self.job_url(&id, "/cancel")).send().await;
                return;
            }
            outcome = self.follow(&id, &tx) => outcome,
        };
        match outcome {
            Ok(result) => match self.store(&id, &result).await {
                Ok(text) => {
                    if tx.send(Chunk::Token(text)).await.is_ok() {
                        let _ = tx.send(Chunk::End(RunEnd::reason("stop"))).await;
                    }
                }
                Err(reason) => {
                    let _ = tx.send(Chunk::Reject(reason)).await;
                }
            },
            Err(reason) => {
                let _ = tx.send(Chunk::Reject(reason)).await;
            }
        }
    }

    fn job_url(&self, id: &str, suffix: &str) -> String {
        format!("{}/sdcpp/v1/jobs/{id}{suffix}", self.url)
    }

    async fn submit(&self) -> Result<String, String> {
        let url = format!("{}/sdcpp/v1/vid_gen", self.url);
        let resp = self
            .client
            .post(&url)
            .json(&self.body)
            .send()
            .await
            .map_err(|e| format!("engine request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(rejection_of(resp).await);
        }
        let accepted: Accepted = resp
            .json()
            .await
            .map_err(|e| format!("engine answered an unreadable job: {e}"))?;
        Ok(accepted.id)
    }

    /// Poll the job to its end, narrating state changes and, while it
    /// generates, the time spent, as reasoning the client may show.
    async fn follow(&self, id: &str, tx: &mpsc::Sender<Chunk>) -> Result<VideoResult, String> {
        let url = self.job_url(id, "");
        let started = tokio::time::Instant::now();
        let mut last_note = String::new();
        let mut last_beat = started;
        loop {
            let resp = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| format!("engine poll failed: {e}"))?;
            if !resp.status().is_success() {
                return Err(format!("engine lost the job: {}", resp.status()));
            }
            let job: JobStatus = resp
                .json()
                .await
                .map_err(|e| format!("engine answered an unreadable job: {e}"))?;
            match job.status.as_str() {
                "completed" => {
                    return job
                        .result
                        .ok_or_else(|| "engine finished with no video".into());
                }
                "failed" => {
                    return Err(job
                        .error
                        .and_then(|e| e.message)
                        .unwrap_or_else(|| "generation failed".into()));
                }
                "cancelled" => return Err("generation cancelled".into()),
                _ => {}
            }
            let note = match job.status.as_str() {
                "queued" => format!("queued, {} ahead", job.queue_position),
                other => other.to_string(),
            };
            let now = tokio::time::Instant::now();
            if note != last_note {
                last_note = note.clone();
                last_beat = now;
                if tx
                    .send(Chunk::Reasoning(format!("{note}\n")))
                    .await
                    .is_err()
                {
                    return Err("client gone".into());
                }
            } else if now.duration_since(last_beat) >= HEARTBEAT {
                last_beat = now;
                let elapsed = now.duration_since(started).as_secs();
                if tx
                    .send(Chunk::Reasoning(format!("{note}, {elapsed}s\n")))
                    .await
                    .is_err()
                {
                    return Err("client gone".into());
                }
            }
            tokio::time::sleep(POLL).await;
        }
    }

    /// Write the clip and return the reply text that points at it.
    async fn store(&self, id: &str, result: &VideoResult) -> Result<String, String> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(result.b64_json.trim())
            .map_err(|e| format!("engine sent an undecodable video: {e}"))?;
        let name = file_name(id, result.output_format.as_deref());
        tokio::fs::create_dir_all(&self.output.dir)
            .await
            .map_err(|e| format!("cannot create {}: {e}", self.output.dir.display()))?;
        let path = self.output.dir.join(&name);
        tokio::fs::write(&path, &bytes)
            .await
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(reply_text(&self.output.link_base, &self.model, &name))
    }
}

/// `{job id}.{format}`, with the id reduced to characters safe in a path
/// and a URL. sd-server's ids are already that; this is the guarantee.
fn file_name(id: &str, format: Option<&str>) -> String {
    let stem: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    let stem = if stem.is_empty() {
        "video".to_string()
    } else {
        stem
    };
    let ext = match format {
        Some("webp") => "webp",
        Some("avi") => "avi",
        _ => "webm",
    };
    format!("{stem}.{ext}")
}

/// The assistant message: a video tag for a client that renders HTML, and
/// a plain link for one that renders only Markdown.
fn reply_text(link_base: &str, model: &str, name: &str) -> String {
    let link = file_link(link_base, model, name);
    format!("<video controls src=\"{link}\"></video>\n\n[{name}]({link})")
}

/// Where the server hands the clip out. `link_base` is joined without a
/// double slash whether or not it ends with one.
pub fn file_link(link_base: &str, model: &str, name: &str) -> String {
    format!(
        "{}/v1/files/{model}/{name}",
        link_base.trim_end_matches('/')
    )
}

/// Whether `name` is a file this engine could have written: one path
/// segment, nothing that could climb out of the output directory.
pub fn is_output_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.starts_with('.')
        && Path::new(name).components().count() == 1
}

async fn rejection_of(resp: reqwest::Response) -> String {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) {
        let msg = value
            .get("error")
            .and_then(|e| e.as_str().or_else(|| e.get("message")?.as_str()))
            .filter(|m| !m.is_empty());
        if let Some(msg) = msg {
            return msg.to_string();
        }
    }
    let body = body.trim();
    if !body.is_empty() {
        let mut s = body.to_string();
        s.truncate(500);
        return s;
    }
    format!("engine returned {status}")
}

#[derive(Deserialize)]
struct Accepted {
    id: String,
}

#[derive(Deserialize)]
struct JobStatus {
    status: String,
    #[serde(default)]
    queue_position: u64,
    #[serde(default)]
    result: Option<VideoResult>,
    #[serde(default)]
    error: Option<JobError>,
}

#[derive(Deserialize)]
struct VideoResult {
    b64_json: String,
    #[serde(default)]
    output_format: Option<String>,
}

#[derive(Deserialize)]
struct JobError {
    #[serde(default)]
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_the_last_user_turn() {
        let msgs = vec![
            ChatMessage::system("be terse"),
            ChatMessage::user("first"),
            ChatMessage::assistant("<video>"),
            ChatMessage::user("  a fox in snow  "),
        ];
        let r = Request::from_messages(&msgs).unwrap();
        assert_eq!(r.prompt, "a fox in snow");
        assert!(r.init_image.is_none());
    }

    #[test]
    fn empty_or_missing_prompt_is_none() {
        assert!(Request::from_messages(&[ChatMessage::user("   ")]).is_none());
        assert!(Request::from_messages(&[ChatMessage::assistant("hi")]).is_none());
    }

    #[test]
    fn attached_image_becomes_the_first_frame() {
        let parts = vec![
            serde_json::json!({"type": "text", "text": "make it move"}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
        ];
        let r = Request::from_messages(&[ChatMessage::user(parts)]).unwrap();
        assert_eq!(r.init_image.as_deref(), Some("data:image/png;base64,AAAA"));
        let body = r.body(&serde_json::json!({"width": 640, "prompt": "ignored"}));
        assert_eq!(body["prompt"], "make it move");
        assert_eq!(body["width"], 640);
        assert_eq!(body["init_image"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn params_that_are_not_a_table_are_ignored() {
        let r = Request::from_messages(&[ChatMessage::user("x")]).unwrap();
        assert_eq!(
            r.body(&serde_json::Value::Null),
            serde_json::json!({"prompt": "x"})
        );
    }

    #[test]
    fn file_names_are_safe_and_carry_the_format() {
        assert_eq!(file_name("job_01HTX", Some("webm")), "job_01HTX.webm");
        assert_eq!(file_name("../etc/passwd", Some("avi")), "etcpasswd.avi");
        assert_eq!(file_name("", None), "video.webm");
        assert!(is_output_name("job_01HTX.webm"));
        assert!(!is_output_name("../x.webm"));
        assert!(!is_output_name(".hidden"));
        assert!(!is_output_name("a/b.webm"));
        assert!(!is_output_name(""));
    }

    #[test]
    fn links_join_without_a_double_slash() {
        assert_eq!(file_link("", "h3", "a.webm"), "/v1/files/h3/a.webm");
        assert_eq!(
            file_link("https://x/api/", "h3", "a.webm"),
            "https://x/api/v1/files/h3/a.webm"
        );
        let text = reply_text("", "h3", "a.webm");
        assert!(text.starts_with("<video controls src=\"/v1/files/h3/a.webm\">"));
        assert!(text.ends_with("[a.webm](/v1/files/h3/a.webm)"));
    }
}
