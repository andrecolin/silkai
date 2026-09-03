//! What the scheduler did and when. A ring of recent events for replay, a
//! broadcast channel for live subscribers, and the same lines mirrored to
//! `tracing`. The ring is shared across config reloads.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::broadcast;

pub const RING: usize = 500;
const CHANNEL: usize = 256;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Event {
    pub seq: u64,
    /// RFC 3339, UTC, millisecond precision.
    pub t: String,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// An event before it gets a sequence number and a timestamp.
#[derive(Debug, Clone, Default)]
pub struct Draft {
    pub kind: &'static str,
    pub model: Option<String>,
    pub job: Option<u64>,
    pub gpu: Option<u32>,
    pub ms: Option<u64>,
    pub error: Option<String>,
}

impl Draft {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            ..Default::default()
        }
    }
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    pub fn job(mut self, job: u64) -> Self {
        self.job = Some(job);
        self
    }
    pub fn gpu(mut self, gpu: u32) -> Self {
        self.gpu = Some(gpu);
        self
    }
    pub fn ms(mut self, ms: u64) -> Self {
        self.ms = Some(ms);
        self
    }
    pub fn error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

pub struct EventLog {
    ring: Mutex<VecDeque<Event>>,
    seq: AtomicU64,
    tx: broadcast::Sender<Event>,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLog {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL);
        Self {
            ring: Mutex::new(VecDeque::with_capacity(RING)),
            seq: AtomicU64::new(0),
            tx,
        }
    }

    pub fn emit(&self, draft: Draft) -> Event {
        let event = Event {
            seq: self.seq.fetch_add(1, Ordering::SeqCst) + 1,
            t: rfc3339_now(),
            kind: draft.kind,
            model: draft.model,
            job: draft.job,
            gpu: draft.gpu,
            ms: draft.ms,
            error: draft.error,
        };
        tracing::info!("{}", line(&event));
        {
            let mut ring = self.ring.lock().expect("event ring");
            if ring.len() == RING {
                ring.pop_front();
            }
            ring.push_back(event.clone());
        }
        let _ = self.tx.send(event.clone());
        event
    }

    /// Events with `seq > after`, oldest first.
    pub fn since(&self, after: u64) -> Vec<Event> {
        self.ring
            .lock()
            .expect("event ring")
            .iter()
            .filter(|e| e.seq > after)
            .cloned()
            .collect()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn last_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }
}

fn line(e: &Event) -> String {
    let mut s = e.kind.to_string();
    if let Some(m) = &e.model {
        s.push(' ');
        s.push_str(m);
    }
    if let Some(j) = e.job {
        s.push_str(&format!(" job={j}"));
    }
    if let Some(g) = e.gpu {
        s.push_str(&format!(" gpu={g}"));
    }
    if let Some(ms) = e.ms {
        s.push_str(&format!(" {ms}ms"));
    }
    if let Some(err) = &e.error {
        s.push_str(&format!(" error={err}"));
    }
    s
}

pub fn rfc3339_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(now.as_secs(), now.subsec_millis())
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm), so
/// the daemon needs no date crate.
pub fn rfc3339(unix_secs: u64, millis: u32) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs = unix_secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_known_instants() {
        assert_eq!(rfc3339(0, 0), "1970-01-01T00:00:00.000Z");
        // 2026-09-03T15:56:11.540Z
        assert_eq!(rfc3339(1_788_450_971, 540), "2026-09-03T15:56:11.540Z");
        // Leap day.
        assert_eq!(rfc3339(1_709_164_800, 0), "2024-02-29T00:00:00.000Z");
    }

    #[test]
    fn ring_keeps_the_newest_and_replays_after_seq() {
        let log = EventLog::new();
        for _ in 0..(RING + 10) {
            log.emit(Draft::new("tick"));
        }
        let all = log.since(0);
        assert_eq!(all.len(), RING);
        assert_eq!(all.first().unwrap().seq, 11);
        assert_eq!(log.since(RING as u64 + 5).len(), 5);
    }

    #[test]
    fn json_omits_empty_fields() {
        let log = EventLog::new();
        let e = log.emit(Draft::new("load").model("write").gpu(0).ms(8210));
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"load\""));
        assert!(json.contains("\"ms\":8210"));
        assert!(!json.contains("job"));
        assert!(!json.contains("error"));
    }
}
