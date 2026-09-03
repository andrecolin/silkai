use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use silkai_adapters::{ChatMessage, RunOptions};
use silkai_sched::JobId;
use tokio::sync::mpsc;

use crate::app::AppState;
use crate::runtime::{Runtime, RuntimeError};

#[derive(Deserialize)]
pub struct SessionQuery {
    pub model: String,
}

/// `{"type":"prompt","content":"..."}` sends one user turn. Send
/// `{"type":"prompt","messages":[{"role":"system","content":"..."},...]}`
/// to give the model a system prompt or history instead. `{"type":"ping"}`
/// keeps the session open without running anything.
#[derive(Deserialize)]
struct ClientMsg {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

impl ClientMsg {
    fn into_prompt(self) -> (Vec<ChatMessage>, RunOptions) {
        let opts = RunOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };
        let messages = if self.messages.is_empty() {
            vec![ChatMessage::user(self.content)]
        } else {
            self.messages
        };
        (messages, opts)
    }
}

#[derive(Serialize)]
struct ServerMsg<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<&'a str>,
}

pub async fn session(
    ws: WebSocketUpgrade,
    Query(q): Query<SessionQuery>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let rt = state.runtime.read().await.clone();
    if !rt.allows_websocket(&q.model) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let model = q.model;
    ws.on_upgrade(move |socket| run_session(socket, rt, model))
        .into_response()
}

async fn run_session(mut socket: WebSocket, rt: Arc<Runtime>, model: String) {
    let _ = send_json(&mut socket, "queued", None, None).await;
    let job = match rt.begin_session(&model).await {
        Ok(job) => job,
        Err(err) => {
            let msg = err.to_string();
            let _ = send_json(&mut socket, "error", None, Some(&msg)).await;
            let _ = socket.send(Message::Close(None)).await;
            return;
        }
    };
    let _ = send_json(&mut socket, "live", None, None).await;
    let idle = rt.idle_timeout(&model);
    loop {
        match recv_or_idle(&mut socket, idle).await {
            Recv::Idle => {
                let _ = send_json(&mut socket, "idle_close", None, None).await;
                break;
            }
            Recv::Keepalive => continue,
            Recv::Closed => break,
            Recv::Prompt(messages, opts) => {
                if let Err(err) =
                    stream_prompt(&mut socket, &rt, job, &model, &messages, &opts).await
                {
                    let msg = err.to_string();
                    let _ = send_json(&mut socket, "error", None, Some(&msg)).await;
                    break;
                }
            }
            Recv::Stop => break,
        }
    }
    rt.end_session(job).await;
    let _ = socket.send(Message::Close(None)).await;
}

enum Recv {
    Prompt(Vec<ChatMessage>, RunOptions),
    Stop,
    /// A ping, a pong, or `{"type":"ping"}`: hold the model on the card and
    /// start the idle timer again. A session that pins a model while the app
    /// talks to the engine directly has no prompts to send, so this is the
    /// only way it can stay open.
    Keepalive,
    Idle,
    Closed,
}

async fn recv_or_idle(socket: &mut WebSocket, idle: Duration) -> Recv {
    match tokio::time::timeout(idle, socket.recv()).await {
        Ok(None) | Ok(Some(Err(_))) => Recv::Closed,
        Ok(Some(Ok(Message::Close(_)))) => Recv::Closed,
        Ok(Some(Ok(Message::Text(text)))) => parse_client(&text),
        // axum answers a ping itself; we only have to not treat it as a close.
        Ok(Some(Ok(Message::Ping(_)))) | Ok(Some(Ok(Message::Pong(_)))) => Recv::Keepalive,
        Ok(Some(Ok(_))) => Recv::Closed,
        Err(_) => Recv::Idle,
    }
}

fn parse_client(text: &str) -> Recv {
    let Ok(msg) = serde_json::from_str::<ClientMsg>(text) else {
        return Recv::Closed;
    };
    match msg.kind.as_str() {
        "prompt" => {
            let (messages, opts) = msg.into_prompt();
            Recv::Prompt(messages, opts)
        }
        "stop" => Recv::Stop,
        "ping" => Recv::Keepalive,
        _ => Recv::Closed,
    }
}

async fn stream_prompt(
    socket: &mut WebSocket,
    rt: &Runtime,
    job: JobId,
    model: &str,
    messages: &[ChatMessage],
    opts: &RunOptions,
) -> Result<(), RuntimeError> {
    let mut rx: mpsc::Receiver<String> = rt.session_prompt(job, model, messages, opts).await?;
    while let Some(text) = rx.recv().await {
        send_json(socket, "token", Some(&text), None).await?;
    }
    send_json(socket, "done", None, None).await?;
    Ok(())
}

async fn send_json(
    socket: &mut WebSocket,
    kind: &str,
    text: Option<&str>,
    message: Option<&str>,
) -> Result<(), RuntimeError> {
    let payload = serde_json::to_string(&ServerMsg {
        kind,
        text,
        message,
    })
    .map_err(|e| RuntimeError::Engine(silkai_adapters::EngineError::Other(e.to_string())))?;
    socket
        .send(Message::Text(payload.into()))
        .await
        .map_err(|e| RuntimeError::Engine(silkai_adapters::EngineError::Other(e.to_string())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_is_a_keepalive_not_a_close() {
        assert!(matches!(
            parse_client(r#"{"type":"ping"}"#),
            Recv::Keepalive
        ));
    }

    #[test]
    fn stop_and_prompt_still_parse() {
        assert!(matches!(parse_client(r#"{"type":"stop"}"#), Recv::Stop));
        assert!(matches!(
            parse_client(r#"{"type":"prompt","content":"hi"}"#),
            Recv::Prompt(..)
        ));
    }

    #[test]
    fn unknown_types_and_junk_still_close() {
        assert!(matches!(parse_client(r#"{"type":"nope"}"#), Recv::Closed));
        assert!(matches!(parse_client("not json"), Recv::Closed));
    }
}
