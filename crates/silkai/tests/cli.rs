use std::path::Path;
use std::process::Command;

fn silkai(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_silkai"))
        .args(args)
        .env_remove("SILKAI_CONFIG")
        .output()
        .expect("run silkai");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn version_and_help() {
    let (code, out, _) = silkai(&["--version"]);
    assert_eq!(code, 0);
    assert_eq!(out.trim(), format!("silkai {}", env!("CARGO_PKG_VERSION")));
    let (code, out, _) = silkai(&["--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("silkai init"));
    assert!(out.contains("silkai check"));
    let (code, _, err) = silkai(&["--nope"]);
    assert_eq!(code, 2);
    assert!(err.contains("unknown argument"));
}

#[test]
fn run_without_config_points_at_init() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("none.toml");
    let (code, _, err) = silkai(&["--config", path.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(err.contains("silkai init"), "{err}");
}

#[test]
fn init_writes_once_unless_forced() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sub").join("config.toml");
    let p = path.to_str().unwrap();
    let (code, out, _) = silkai(&["init", "--config", p]);
    assert_eq!(code, 0, "{out}");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("[resources]"));
    assert!(text.contains("[models.chat]"));
    assert!(text.contains("CHANGE-ME.gguf"));
    let (code, _, err) = silkai(&["init", "--config", p]);
    assert_eq!(code, 1);
    assert!(err.contains("--force"));
    let (code, _, _) = silkai(&["init", "--force", "--config", p]);
    assert_eq!(code, 0);
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    path.to_str().unwrap().to_string()
}

const RESOURCES: &str = "[resources]\ngpu_total_gb = 32\ngpu_headroom_gb = 3\nram_total_gb = 128\nram_headroom_gb = 32\n";

#[test]
fn check_passes_a_clean_fake_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "ok.toml",
        &format!(
            "{RESOURCES}\n[models.a]\nengine = \"fake\"\npath = \"/x\"\nvram_gb = 8\npriority = \"live\"\n\n[models.huge]\nengine = \"fake\"\npath = \"/y\"\nvram_gb = 40\npriority = \"normal\"\n"
        ),
    );
    let (code, out, err) = silkai(&["check", "--config", &p]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("GPU 0: 29.0 GB schedulable"));
    assert!(out.contains("RAM shelf: 96.0 GB"));
    assert!(out.contains("ok"));
    assert!(out.contains("huge") && out.contains("DISABLED"));
}

#[test]
fn check_reports_missing_command_and_files() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "bad.toml",
        &format!(
            "{RESOURCES}\n[models.chat]\nengine = \"process\"\npath = \"chat\"\nurl = \"http://127.0.0.1:8101\"\ncmd = [\"/no/such/llama-server\", \"--model\", \"/no/such/model.gguf\"]\nvram_gb = 8\npriority = \"normal\"\n"
        ),
    );
    let (code, out, err) = silkai(&["check", "--config", &p]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("command not found: /no/such/llama-server"),
        "{out}"
    );
    assert!(out.contains("missing file: /no/such/model.gguf"), "{out}");
    assert!(err.contains("2 problems"), "{err}");
}

#[test]
fn check_rejects_broken_toml() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "broken.toml", "this is not toml [[[");
    let (code, _, err) = silkai(&["check", "--config", &p]);
    assert_eq!(code, 1);
    assert!(!err.is_empty());
}
