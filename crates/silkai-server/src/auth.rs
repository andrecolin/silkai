//! One shared token for the page, the metrics, and the admin routes. Sent as
//! `Authorization: Bearer <token>` by tools, or as HTTP Basic (any user, the
//! token as password) by a browser, which cannot add a header to a
//! navigation but does know how to ask for a password.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::app::AppState;

pub(crate) async fn require_token(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let Some(token) = state.ui.token.as_deref() else {
        return next.run(req).await;
    };
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    if presented.as_deref().is_some_and(|h| authorized(h, token)) {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(
            header::WWW_AUTHENTICATE,
            r#"Basic realm="silkai", charset="UTF-8""#,
        )],
        "token required",
    )
        .into_response()
}

fn authorized(header: &str, token: &str) -> bool {
    if let Some(bearer) = header.strip_prefix("Bearer ") {
        return constant_time_eq(bearer.trim().as_bytes(), token.as_bytes());
    }
    if let Some(basic) = header.strip_prefix("Basic ") {
        let Some(decoded) = base64_decode(basic.trim()) else {
            return false;
        };
        let Ok(text) = std::str::from_utf8(&decoded) else {
            return false;
        };
        // "user:password"; the user part is ignored.
        let password = text.split_once(':').map(|(_, p)| p).unwrap_or(text);
        return constant_time_eq(password.as_bytes(), token.as_bytes());
    }
    false
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => return None,
        };
        buf = (buf << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_and_basic_both_work() {
        assert!(authorized("Bearer s3cret", "s3cret"));
        assert!(!authorized("Bearer s3cre", "s3cret"));
        // "silkai:s3cret"
        assert!(authorized("Basic c2lsa2FpOnMzY3JldA==", "s3cret"));
        assert!(!authorized("Basic c2lsa2FpOm5vcGU=", "s3cret"));
        assert!(!authorized("Digest x", "s3cret"));
    }

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGk").unwrap(), b"hi");
        assert!(base64_decode("not base64!").is_none());
    }
}
