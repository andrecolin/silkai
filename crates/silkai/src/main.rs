fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let path = std::env::var("SILKAI_CONFIG").unwrap_or_else(|_| default_config_path());
    let cfg = silkai_server::config::load_from_path(&path)?;
    tokio::runtime::Runtime::new()?.block_on(silkai_server::serve(cfg))
}

fn default_config_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{home}/.config/silkai/config.toml")
}
