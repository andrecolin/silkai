//! `silkai`: a local GPU scheduler daemon. Several models, one card; what
//! fits runs together, the rest waits or parks. Configuration lives in
//! `~/.config/silkai/config.toml` or `$SILKAI_CONFIG`.
//!
//! ```text
//! silkai                  run the daemon
//! silkai init             write a starter config for this machine
//! silkai check            validate the config and every path in it
//! silkai --config PATH    use another config file (any command)
//! ```

use std::path::{Path, PathBuf};

use silkai_server::config::{self, ConfiguredModel};

const USAGE: &str = "\
silkai: run several models on one GPU; what fits runs together, the rest waits or parks.

Usage:
  silkai [--config PATH]            run the daemon
  silkai init [--force] [--config PATH]
                                    write a starter config for this machine
  silkai check [--config PATH]      load the config, print the plan, verify every path
  silkai --version | --help

Config: --config, else $SILKAI_CONFIG, else ~/.config/silkai/config.toml.
";

enum Command {
    Run,
    Init { force: bool },
    Check,
    Help,
    Version,
}

struct Cli {
    command: Command,
    config: PathBuf,
}

fn main() {
    let cli = match parse(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let result = match cli.command {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Version => {
            println!("silkai {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Init { force } => init(&cli.config, force),
        Command::Check => check(&cli.config),
        Command::Run => run(&cli.config),
    };
    if let Err(err) = result {
        eprintln!("silkai: {err:#}");
        std::process::exit(1);
    }
}

fn parse(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut command = Command::Run;
    let mut config = None;
    let mut force = false;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" | "help" => command = Command::Help,
            "-V" | "--version" | "version" => command = Command::Version,
            "init" => command = Command::Init { force },
            "check" => command = Command::Check,
            "run" => command = Command::Run,
            "--force" => force = true,
            "-c" | "--config" => {
                config = Some(PathBuf::from(args.next().ok_or("--config needs a path")?));
            }
            other if other.starts_with("--config=") => {
                config = Some(PathBuf::from(&other["--config=".len()..]));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if let Command::Init { force: f } = &mut command {
        *f = force;
    }
    Ok(Cli {
        command,
        config: config.unwrap_or_else(default_config_path),
    })
}

fn default_config_path() -> PathBuf {
    if let Ok(path) = std::env::var("SILKAI_CONFIG") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(format!("{home}/.config/silkai/config.toml"))
}

fn run(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!(
            "no config at {}. Run `silkai init` to write one for this machine.",
            path.display()
        );
    }
    tracing_subscriber::fmt::init();
    let cfg = config::load_from_path(path)?;
    tokio::runtime::Runtime::new()?.block_on(silkai_server::serve(cfg, Some(path.to_path_buf())))
}

// ---------------------------------------------------------------- init

fn init(path: &Path, force: bool) -> anyhow::Result<()> {
    if path.exists() && !force {
        anyhow::bail!(
            "{} already exists; pass --force to overwrite it",
            path.display()
        );
    }
    let gpus = config::probe_gpus().unwrap_or_default();
    let ram = config::probe_ram_gb();
    let server = find_on_path("llama-server");
    let text = starter_config(&gpus, ram, server.as_deref());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)?;
    println!("Wrote {}", path.display());
    match (&gpus[..], ram) {
        ([], _) => {
            println!("No NVIDIA GPU found by nvidia-smi: set resources.gpu_total_gb by hand.")
        }
        (cards, Some(ram)) => println!(
            "Probed {} GPU{} ({}) and {ram:.0} GB RAM.",
            cards.len(),
            if cards.len() == 1 { "" } else { "s" },
            cards
                .iter()
                .map(|(i, gb)| format!("#{i} {gb:.0} GB"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        (cards, None) => println!(
            "Probed {} GPU(s); RAM unknown, set resources.ram_total_gb.",
            cards.len()
        ),
    }
    match server {
        Some(p) => println!("llama-server found at {}.", p.display()),
        None => {
            println!("llama-server not on PATH: build llama.cpp, or edit `cmd` to point at it.")
        }
    }
    println!("Next: edit the model path in the file, then `silkai check`.");
    Ok(())
}

fn starter_config(gpus: &[(u32, f64)], ram: Option<f64>, server: Option<&Path>) -> String {
    let mut out = String::new();
    out.push_str("# SilkAI config written by `silkai init`. Edit the GGUF path, then run\n");
    out.push_str("# `silkai check`, then `silkai`. Docs: https://github.com/andrecolin/silkai\n\n");
    out.push_str("listen = \"127.0.0.1:8080\"\n\n");
    out.push_str("[resources]\n");
    match gpus {
        [] => {
            out.push_str("# nvidia-smi found no card. Set this to your GPU's memory in GB.\n");
            out.push_str("gpu_total_gb = 24\n");
        }
        [(_, gb)] => {
            out.push_str(&format!(
                "# Probed one GPU with {gb:.0} GB; the daemon re-probes at start.\n"
            ));
        }
        many => {
            out.push_str(&format!(
                "# Probed {} GPUs; the daemon re-probes at start.\n",
                many.len()
            ));
        }
    }
    out.push_str("gpu_headroom_gb = 3          # left free for the desktop and the driver\n");
    match ram {
        Some(gb) => out.push_str(&format!(
            "# Probed {gb:.0} GB RAM; the daemon re-probes at start.\n"
        )),
        None => out
            .push_str("ram_total_gb = 32           # RAM probe failed; set your machine's total\n"),
    }
    out.push_str("ram_headroom_gb = 16         # left for the OS and other programs\n");
    out.push_str("prefetch_on_start = true\n");
    out.push_str("request_timeout_secs = 600\n\n");
    out.push_str(
        "# Optional status page at http://127.0.0.1:8080/ui\n# [ui]\n# enabled = true\n\n",
    );
    let bin = server
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "llama-server".into());
    out.push_str("# One chat model as a llama-server child that SilkAI starts and stops.\n");
    out.push_str("# `path` is the name clients send; `--model` is the GGUF on disk.\n");
    out.push_str("# Set vram_gb to what the model takes on the card (weights plus context).\n");
    out.push_str(
        "[models.chat]\nengine = \"process\"\npath = \"chat\"\nurl = \"http://127.0.0.1:8101\"\n",
    );
    out.push_str(&format!(
        "cmd = [\"{bin}\", \"--model\", \"/models/CHANGE-ME.gguf\",\n       \"--alias\", \"chat\", \"--host\", \"127.0.0.1\", \"--port\", \"8101\",\n       \"--n-gpu-layers\", \"999\", \"--jinja\", \"--no-webui\"]\n"
    ));
    out.push_str("vram_gb = 8\npriority = \"normal\"\nkeep_warm = true\ntransport = \"http\"\n");
    out
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

// ---------------------------------------------------------------- check

fn check(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!(
            "no config at {}. Run `silkai init` to write one.",
            path.display()
        );
    }
    let cfg = config::load_from_path(path)?;
    println!("config: {}", path.display());
    for gpu in cfg.resources.benches() {
        println!("GPU {}: {:.1} GB schedulable", gpu.id, gpu.schedulable_gb);
    }
    println!("RAM shelf: {:.1} GB", cfg.resources.ram_shelf_gb);
    println!("listen: {}", cfg.listen);
    let mut problems = 0;
    for model in &cfg.enabled {
        let issues = model_issues(model);
        problems += issues.len();
        print_model(model, &issues);
    }
    for model in &cfg.disabled {
        println!(
            "  {:<12} {:<10} {:>5.1} GB  DISABLED: larger than any card's schedulable memory",
            model.spec.name, model.engine, model.spec.vram_gb
        );
    }
    if cfg.enabled.is_empty() {
        println!("no models configured");
        problems += 1;
    }
    let ui = if cfg.ui.enabled { "on" } else { "off" };
    println!("status page: {ui}");
    if problems > 0 {
        anyhow::bail!(
            "{problems} problem{} found",
            if problems == 1 { "" } else { "s" }
        );
    }
    println!("ok");
    Ok(())
}

fn print_model(model: &ConfiguredModel, issues: &[String]) {
    let flags = format!(
        "{}{}",
        format!("{:?}", model.spec.priority).to_lowercase(),
        if model.spec.exclusive {
            " exclusive"
        } else {
            ""
        }
    );
    let verdict = if issues.is_empty() {
        "ok".to_string()
    } else {
        issues.join("; ")
    };
    println!(
        "  {:<12} {:<10} {:>5.1} GB  {:<20} {verdict}",
        model.spec.name, model.engine, model.spec.vram_gb, flags
    );
}

fn model_issues(model: &ConfiguredModel) -> Vec<String> {
    let mut issues = Vec::new();
    match model.engine.as_str() {
        "process" => {
            match model.cmd.first() {
                None => issues.push("cmd is empty".into()),
                Some(bin) if !executable_exists(bin) => {
                    issues.push(format!("command not found: {bin}"));
                }
                _ => {}
            }
            for arg in model.cmd.iter().skip(1) {
                if looks_like_file(arg) && !Path::new(arg).exists() {
                    issues.push(format!("missing file: {arg}"));
                }
            }
            check_url(model, &mut issues);
        }
        "llama.cpp" => {
            if !Path::new(&model.path).exists() {
                issues.push(format!("missing file: {}", model.path));
            }
            if !cfg!(feature = "llama") {
                issues.push("this binary was built without --features llama".into());
            }
        }
        "vllm" | "ollama" => check_url(model, &mut issues),
        "fake" => {}
        other => issues.push(format!("unknown engine: {other}")),
    }
    issues
}

fn check_url(model: &ConfiguredModel, issues: &mut Vec<String>) {
    if let Some(url) = &model.url {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            issues.push(format!("url must start with http:// or https://: {url}"));
        }
    }
}

fn executable_exists(bin: &str) -> bool {
    if bin.contains('/') {
        return Path::new(bin).is_file();
    }
    find_on_path(bin).is_some()
}

/// Arguments that name a file on disk: absolute paths with an extension.
fn looks_like_file(arg: &str) -> bool {
    arg.starts_with('/') && Path::new(arg).extension().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands_and_config_flag() {
        let cli = parse(
            ["init", "--force", "--config", "/tmp/x.toml"]
                .map(String::from)
                .into_iter(),
        )
        .unwrap();
        assert!(matches!(cli.command, Command::Init { force: true }));
        assert_eq!(cli.config, PathBuf::from("/tmp/x.toml"));
        let cli = parse(
            ["check", "--config=/tmp/y.toml"]
                .map(String::from)
                .into_iter(),
        )
        .unwrap();
        assert!(matches!(cli.command, Command::Check));
        assert_eq!(cli.config, PathBuf::from("/tmp/y.toml"));
        assert!(parse(["--bogus"].map(String::from).into_iter()).is_err());
        assert!(parse(["--config"].map(String::from).into_iter()).is_err());
    }

    #[test]
    fn starter_config_parses_in_every_probe_outcome() {
        for (gpus, ram) in [
            (vec![], None),
            (vec![(0, 32.0)], Some(125.0)),
            (vec![(0, 24.0), (1, 24.0)], Some(64.0)),
        ] {
            let text = starter_config(&gpus, ram, Some(Path::new("/usr/local/bin/llama-server")));
            // With no probe the file must carry explicit totals so it loads
            // on a box without nvidia-smi.
            let probed = config::load_from_str_probed_ram(&text, ram.unwrap_or(32.0));
            if gpus.is_empty() {
                assert!(probed.is_ok(), "{text}");
            }
            assert!(text.contains("[models.chat]"));
        }
    }

    #[test]
    fn file_heuristic() {
        assert!(looks_like_file("/models/x.gguf"));
        assert!(!looks_like_file("--model"));
        assert!(!looks_like_file("/usr/bin/llama-server"));
        assert!(!looks_like_file("127.0.0.1"));
    }
}
