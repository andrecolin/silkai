//! Engines SilkAI can place on a card: a managed child process speaking
//! OpenAI chat (llama-server, vLLM), HTTP adapters for vLLM and Ollama, an
//! optional in-process llama.cpp behind `--features llama`, and a fake for
//! tests. The [`Engine`] trait is what the runtime drives: warm, load, wake,
//! sleep, discard, run.

mod fake;
mod llama;
mod ollama;
mod process;
mod vllm;
pub use fake::FakeEngine;
pub use llama::LlamaEngine;
pub use ollama::OllamaEngine;
pub use process::ProcessEngine;
pub use vllm::VllmEngine;

use std::borrow::Cow;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// What one turn says. A client sends either a plain string or a list of
/// OpenAI content parts, and a list may hold an image alongside the text.
/// Both shapes are kept as they arrived and serialized back unchanged, so an
/// engine that understands images is handed them. [`Content::text`] is the
/// projection for engines whose wire format has no place for parts.
///
/// Untagged: a JSON string deserializes to `Text`, a JSON array to `Parts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

impl Content {
    /// This content as plain text: the string itself, or the `text` fields of
    /// the parts joined in order. A part carrying no text — an image — adds
    /// nothing, so a message that is only an image projects to `""`.
    pub fn text(&self) -> Cow<'_, str> {
        match self {
            Content::Text(s) => Cow::Borrowed(s),
            Content::Parts(parts) => Cow::Owned(
                parts
                    .iter()
                    .filter_map(|p| p.get("text")?.as_str())
                    .collect::<Vec<_>>()
                    .join(""),
            ),
        }
    }

    /// Whether this is a list of parts rather than a plain string.
    pub fn is_parts(&self) -> bool {
        matches!(self, Content::Parts(_))
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

impl From<Vec<serde_json::Value>> for Content {
    fn from(parts: Vec<serde_json::Value>) -> Self {
        Content::Parts(parts)
    }
}

/// One turn of an OpenAI-style chat: `system`, `user`, `assistant`, or
/// whatever role the engine's template understands. The whole list reaches
/// the engine; SilkAI never collapses it to a single string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Content,
    /// The tool calls an assistant turn asked for, verbatim. Carried so a
    /// tool-using conversation can be replayed to the engine: an assistant
    /// turn that requested a call, and the `tool` turn answering it, are both
    /// part of the history the next request must reproduce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Which call a `role: "tool"` turn is answering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on some providers' tool turns; passed through untouched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<Content>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn system(content: impl Into<Content>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<Content>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Self::new("assistant", content)
    }
}

/// What an engine sends back while it runs: the text as it arrives, then at
/// most one `End` saying how the run finished. An engine that reports
/// neither a reason nor a count sends only `Token`s, and the server falls
/// back to `"stop"` with no usage — what every reply carried before engines
/// could say otherwise.
#[derive(Debug, Clone, PartialEq)]
pub enum Chunk {
    Token(String),
    /// A chunk of the engine's reasoning trace, when it reports one apart
    /// from the answer. Carried separately so it never lands in the answer
    /// text: it is shown live and then discarded, not persisted.
    Reasoning(String),
    /// One `delta.tool_calls` fragment, exactly as the engine sent it.
    /// Fragments carry an `index` and a partial `arguments` string that the
    /// receiver assembles; forwarding them verbatim keeps that contract
    /// between the client and the engine rather than reinterpreting it.
    ToolCalls(serde_json::Value),
    End(RunEnd),
    /// The engine refused this request (context window exceeded, and so on).
    /// The string is the engine's own reason and ends the stream without a
    /// token or an `End`.
    Reject(String),
}

impl Chunk {
    /// The text of a token chunk. `End` carries none.
    pub fn text(&self) -> Option<&str> {
        match self {
            Chunk::Token(t) => Some(t),
            // Neither reasoning nor a tool call is answer text.
            Chunk::Reasoning(_) => None,
            Chunk::ToolCalls(_) => None,
            Chunk::End(_) => None,
            Chunk::Reject(_) => None,
        }
    }
}

/// What the engine said once generation stopped. Both fields are optional:
/// an engine reports what it knows and nothing more.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunEnd {
    /// The engine's own reason — `"stop"`, `"length"`, and so on. Passed
    /// through rather than interpreted, since only the engine knows why it
    /// stopped.
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

impl RunEnd {
    pub fn reason(reason: impl Into<String>) -> Self {
        Self {
            finish_reason: Some(reason.into()),
            usage: None,
        }
    }
}

/// Tokens counted for one run, as the engine counted them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Per-request generation settings, taken from the OpenAI-style body.
/// `None` means the engine's own default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// The client's tool declarations and choice, forwarded verbatim. silkai
    /// does not model the tool schema — it is the engine's contract with the
    /// client, and anything silkai reshapes here it would eventually drop.
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
}

/// The text a plain completion engine sees: the last message's content.
/// Used by engines that have no chat template of their own.
pub fn last_content(messages: &[ChatMessage]) -> Cow<'_, str> {
    messages
        .last()
        .map(|m| m.content.text())
        .unwrap_or(Cow::Borrowed(""))
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not loaded")]
    NotLoaded,
    /// The request itself is unusable (too long for the context window,
    /// and so on). The engine is fine; only this job fails.
    #[error("{0}")]
    Rejected(String),
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait Engine: Send + Sync {
    async fn warm(&self, path: &str) -> Result<(), EngineError>;
    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError>;
    async fn wake(&self, gpu: u32) -> Result<(), EngineError>;
    async fn sleep(&self) -> Result<(), EngineError>;
    async fn discard(&self) -> Result<(), EngineError>;
    /// Generate for `messages`. `prefix` is text already streamed to the
    /// client by an earlier, preempted run; the engine continues after it
    /// and must not emit it again.
    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<Chunk>, EngineError>;
    fn measured_vram_gb(&self) -> f64;

    /// Whether `sleep` keeps a copy in host RAM that `wake` restores without
    /// touching disk. Engines that kill a child or re-read the file say no,
    /// so status does not report RAM that is not held.
    fn has_shelf(&self) -> bool {
        false
    }

    /// The OS process holding this model's VRAM, if any, so the sampler can
    /// attribute what the card measures.
    fn pid(&self) -> Option<u32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_content_borrows_its_string() {
        let c = Content::Text("hello".into());
        assert_eq!(c.text(), "hello");
        assert!(!c.is_parts());
    }

    #[test]
    fn parts_project_to_their_joined_text() {
        let c = Content::Parts(vec![
            serde_json::json!({"type": "text", "text": "hel"}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "data:,"}}),
            serde_json::json!({"type": "text", "text": "lo"}),
        ]);
        assert_eq!(c.text(), "hello");
        assert!(c.is_parts());
    }

    /// A message that is only an image has no text to give a plain-string
    /// engine, and must not panic reaching for one.
    #[test]
    fn image_only_parts_project_to_empty() {
        let c = Content::Parts(vec![serde_json::json!({"type": "image_url"})]);
        assert_eq!(c.text(), "");
    }

    #[test]
    fn string_and_list_round_trip_unchanged() {
        for raw in [
            r#"{"role":"user","content":"hello"}"#,
            r#"{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:,"}}]}"#,
        ] {
            let m: ChatMessage = serde_json::from_str(raw).unwrap();
            let back = serde_json::to_value(&m).unwrap();
            assert_eq!(
                back,
                serde_json::from_str::<serde_json::Value>(raw).unwrap()
            );
        }
    }

    #[test]
    fn last_content_reads_the_final_turn() {
        let msgs = vec![ChatMessage::system("be terse"), ChatMessage::user("hello")];
        assert_eq!(last_content(&msgs), "hello");
        assert_eq!(last_content(&[]), "");
    }
}
