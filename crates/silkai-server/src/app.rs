use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{ModelSpec, Priority, StatusSnapshot};

use crate::config::{AppConfig, ConfiguredModel};
use crate::runtime::Runtime;

pub async fn app_from_config(cfg: AppConfig) -> Router {
    let rt = Runtime::new(cfg).await.expect("runtime");
    router(Arc::new(rt))
}

pub async fn test_app() -> Router {
    app_from_config(clinic_cfg()).await
}

pub async fn test_app_with_disabled() -> Router {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    app_from_config(cfg).await
}

fn router(rt: Arc<Runtime>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/status", get(status))
        .with_state(rt)
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(rt): State<Arc<Runtime>>) -> Json<ModelList> {
    Json(ModelList::from_names(rt.configured_models()))
}

async fn status(State(rt): State<Arc<Runtime>>) -> Json<StatusSnapshot> {
    Json(rt.status())
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
}

impl ModelList {
    fn from_names(names: Vec<String>) -> Self {
        Self {
            object: "list",
            data: names.into_iter().map(ModelEntry::new).collect(),
        }
    }
}

impl ModelEntry {
    fn new(id: String) -> Self {
        Self {
            id,
            object: "model",
        }
    }
}

fn clinic_cfg() -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        resources: clinic_resources(),
        enabled: clinic_models().into_iter().map(fake_model).collect(),
        disabled: vec![],
    }
}

fn too_big() -> ConfiguredModel {
    fake_model(ModelSpec {
        name: "too-big".into(),
        vram_gb: 40.0,
        ram_gb: 40.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
    })
}

fn fake_model(spec: ModelSpec) -> ConfiguredModel {
    ConfiguredModel {
        engine: "fake".into(),
        path: format!("/models/{}.bin", spec.name),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec,
    }
}
