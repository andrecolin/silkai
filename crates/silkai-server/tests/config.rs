use silkai_sched::Priority;
use silkai_server::config::load_from_str;

const TOML: &str = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_headroom_gb = 32
prefetch_on_start = true
request_timeout_secs = 600
ram_total_gb = 128

[models.whisper]
engine = "fake"
path = "/models/whisper.bin"
vram_gb = 12
priority = "live"
exclusive = false
slots = 2
keep_warm = true
transport = "websocket"
idle_timeout_secs = 45

[models.soap]
engine = "llama.cpp"
path = "/models/soap-q4.gguf"
vram_gb = 28
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"

[models.too-big]
engine = "fake"
path = "/models/huge.gguf"
vram_gb = 40
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"
"#;

#[test]
fn parses_clinic_and_disables_too_big() {
    let cfg = load_from_str(TOML).unwrap();
    assert_eq!(cfg.listen, "127.0.0.1:8080");
    assert_eq!(cfg.resources.gpu_schedulable_gb, 29.0);
    assert_eq!(cfg.resources.ram_shelf_gb, 96.0);
    assert!(cfg.prefetch_on_start);
    assert_eq!(cfg.request_timeout_secs, 600);
    let soap = cfg.enabled.iter().find(|m| m.spec.name == "soap").unwrap();
    assert_eq!(soap.spec.priority, Priority::Normal);
    assert!(soap.spec.exclusive);
    assert_eq!(soap.engine, "llama.cpp");
    assert!(cfg.disabled.iter().any(|m| m.spec.name == "too-big"));
}

#[test]
fn invalid_toml_errors() {
    assert!(load_from_str("listen = ").is_err());
}

#[test]
fn missing_gpu_total_errors() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_headroom_gb = 3
ram_headroom_gb = 32
ram_total_gb = 128
"#;
    assert!(load_from_str(t).is_err());
}
