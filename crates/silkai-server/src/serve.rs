use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use tokio::net::TcpListener;

use crate::app::app_from_config_path;
use crate::config::AppConfig;

pub async fn serve(cfg: AppConfig, config_path: Option<PathBuf>) -> anyhow::Result<()> {
    let listener = TcpListener::bind(local_listen_addr(&cfg.listen)?).await?;
    serve_listener(listener, cfg, config_path).await
}

pub async fn serve_listener(
    listener: TcpListener,
    cfg: AppConfig,
    config_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    axum::serve(listener, app_from_config_path(cfg, config_path).await).await?;
    Ok(())
}

fn local_listen_addr(listen: &str) -> anyhow::Result<SocketAddr> {
    let addr = parse_listen(listen)?;
    require_localhost(addr)?;
    Ok(addr)
}

fn parse_listen(listen: &str) -> anyhow::Result<SocketAddr> {
    listen.parse().or_else(|_| parse_localhost_name(listen))
}

fn parse_localhost_name(listen: &str) -> anyhow::Result<SocketAddr> {
    let (host, port) = host_port(listen)?;
    anyhow::ensure!(
        host.eq_ignore_ascii_case("localhost"),
        "listen host must be 127.0.0.1 or localhost"
    );
    Ok(SocketAddr::from((Ipv4Addr::LOCALHOST, parse_port(port)?)))
}

fn host_port(listen: &str) -> anyhow::Result<(&str, &str)> {
    listen
        .rsplit_once(':')
        .filter(|(host, port)| !host.is_empty() && !port.is_empty())
        .ok_or_else(|| anyhow::anyhow!("invalid listen address: {listen}"))
}

fn parse_port(port: &str) -> anyhow::Result<u16> {
    port.parse()
        .map_err(|_| anyhow::anyhow!("invalid listen port: {port}"))
}

fn require_localhost(addr: SocketAddr) -> anyhow::Result<()> {
    anyhow::ensure!(
        addr.ip() == IpAddr::V4(Ipv4Addr::LOCALHOST),
        "listen host must be 127.0.0.1 or localhost"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_ok() {
        assert_eq!(
            local_listen_addr("127.0.0.1:8080").unwrap(),
            "127.0.0.1:8080".parse().unwrap()
        );
    }

    #[test]
    fn localhost_name_ok() {
        assert_eq!(
            local_listen_addr("localhost:8080").unwrap(),
            "127.0.0.1:8080".parse().unwrap()
        );
    }

    #[test]
    fn wildcard_rejected() {
        assert!(local_listen_addr("0.0.0.0:8080").is_err());
    }

    #[test]
    fn public_ip_rejected() {
        assert!(local_listen_addr("8.8.8.8:8080").is_err());
    }

    #[test]
    fn ipv6_loopback_rejected() {
        assert!(local_listen_addr("[::1]:8080").is_err());
    }
}
