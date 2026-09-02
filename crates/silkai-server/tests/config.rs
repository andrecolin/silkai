use silkai_sched::Priority;
use silkai_server::config::{
    load_from_str, load_from_str_probed, load_from_str_probed_ram, parse_meminfo, parse_nvidia_smi,
};

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
fn parses_two_gpus_and_pin() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
ram_total_gb = 128
ram_headroom_gb = 32

[[resources.gpus]]
id = 0
total_gb = 32
headroom_gb = 3

[[resources.gpus]]
id = 1
total_gb = 32
headroom_gb = 3

[models.write]
engine = "fake"
path = "/models/write.gguf"
vram_gb = 26
priority = "normal"
exclusive = true

[models.index]
engine = "fake"
path = "/models/index.gguf"
vram_gb = 10
priority = "background"
gpu = 1
"#;
    let cfg = load_from_str(t).unwrap();
    assert_eq!(cfg.resources.gpus.len(), 2);
    assert_eq!(cfg.resources.max_schedulable(), 29.0);
    let index = cfg.enabled.iter().find(|m| m.spec.name == "index").unwrap();
    assert_eq!(index.spec.gpu, Some(1));
    assert!(cfg.enabled.iter().any(|m| m.spec.name == "write"));
}

#[test]
fn parses_vllm_engine_and_url() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.write]
engine = "vllm"
path = "Qwen/Qwen3-0.6B"
url = "http://127.0.0.1:9000"
vram_gb = 28
priority = "normal"
exclusive = true
"#;
    let cfg = load_from_str(t).unwrap();
    let write = cfg.enabled.iter().find(|m| m.spec.name == "write").unwrap();
    assert_eq!(write.engine, "vllm");
    assert_eq!(write.path, "Qwen/Qwen3-0.6B");
    assert_eq!(write.url.as_deref(), Some("http://127.0.0.1:9000"));
}

#[test]
fn parses_ollama_engine_and_url() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.write]
engine = "ollama"
path = "llama3.2"
url = "http://127.0.0.1:11434"
vram_gb = 8
priority = "normal"
exclusive = true
"#;
    let cfg = load_from_str(t).unwrap();
    let write = cfg.enabled.iter().find(|m| m.spec.name == "write").unwrap();
    assert_eq!(write.engine, "ollama");
    assert_eq!(write.path, "llama3.2");
    assert_eq!(write.url.as_deref(), Some("http://127.0.0.1:11434"));
}

#[test]
fn parses_process_engine_cmd() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.write]
engine = "process"
path = "Qwen/Qwen3-0.6B"
url = "http://127.0.0.1:8001"
cmd = ["vllm", "serve", "Qwen/Qwen3-0.6B", "--port", "8001"]
vram_gb = 28
priority = "normal"
exclusive = true
"#;
    let cfg = load_from_str(t).unwrap();
    let write = cfg.enabled.iter().find(|m| m.spec.name == "write").unwrap();
    assert_eq!(write.engine, "process");
    assert_eq!(
        write.cmd,
        vec!["vllm", "serve", "Qwen/Qwen3-0.6B", "--port", "8001"]
    );
    assert_eq!(write.url.as_deref(), Some("http://127.0.0.1:8001"));
}

#[test]
fn process_engine_without_cmd_errors() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_total_gb = 32
ram_total_gb = 128
[models.write]
engine = "process"
path = "Qwen/Qwen3-0.6B"
vram_gb = 8
priority = "normal"
"#;
    assert!(load_from_str(t).is_err());
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

#[test]
fn parse_nvidia_smi_two_cards_mib() {
    let gpus = parse_nvidia_smi("0, 32768\n1, 24576\n").unwrap();
    assert_eq!(gpus, vec![(0, 32.0), (1, 24.0)]);
}

#[test]
fn probe_fills_gpus_when_totals_omitted() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.write]
engine = "fake"
path = "/models/write.gguf"
vram_gb = 26
priority = "normal"
exclusive = true
"#;
    let cfg = load_from_str_probed(t, vec![(0, 32.0), (1, 32.0)]).unwrap();
    assert_eq!(cfg.resources.gpus.len(), 2);
    assert_eq!(cfg.resources.gpus[0].id, 0);
    assert_eq!(cfg.resources.gpus[0].schedulable_gb, 29.0);
    assert_eq!(cfg.resources.gpus[1].schedulable_gb, 29.0);
    assert!(cfg.enabled.iter().any(|m| m.spec.name == "write"));
}

#[test]
fn explicit_gpu_total_ignores_probe() {
    let t = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 16
gpu_headroom_gb = 1
ram_total_gb = 64
ram_headroom_gb = 8
"#;
    let cfg = load_from_str_probed(t, vec![(0, 80.0)]).unwrap();
    assert!(cfg.resources.gpus.is_empty());
    assert_eq!(cfg.resources.gpu_schedulable_gb, 15.0);
}

#[test]
fn parse_meminfo_kib_to_gb() {
    let text = "MemTotal:       134217728 kB\nMemFree:        1000 kB\n";
    assert_eq!(parse_meminfo(text).unwrap(), 128.0);
}

#[test]
fn missing_ram_total_errors_without_probe() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_headroom_gb = 32
"#;
    assert!(load_from_str(t).is_err());
}

#[test]
fn probe_fills_ram_when_omitted() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_headroom_gb = 32
"#;
    let cfg = load_from_str_probed_ram(t, 128.0).unwrap();
    assert_eq!(cfg.resources.ram_shelf_gb, 96.0);
}

#[test]
fn explicit_ram_total_ignores_probe() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_total_gb = 16
gpu_headroom_gb = 1
ram_total_gb = 64
ram_headroom_gb = 8
"#;
    let cfg = load_from_str_probed_ram(t, 256.0).unwrap();
    assert_eq!(cfg.resources.ram_shelf_gb, 56.0);
}
