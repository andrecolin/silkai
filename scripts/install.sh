#!/usr/bin/env bash
# Install the silkai binary to PREFIX/bin (default: ~/.local/bin)
# and a user systemd unit on Linux.
#
#   ./scripts/install.sh
#   FEATURES=llama ./scripts/install.sh
#   FEATURES=llama,cuda ./scripts/install.sh
#   FEATURES=llama,vulkan ./scripts/install.sh
#   FEATURES=llama,metal ./scripts/install.sh
#   PREFIX=/usr/local ./scripts/install.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
prefix="${PREFIX:-$HOME/.local}"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/silkai"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
bin="$prefix/bin/silkai"

cd "$root"

if [[ -n "${FEATURES:-}" ]]; then
  cargo build --release -p silkai --features "$FEATURES"
else
  cargo build --release -p silkai
fi

mkdir -p "$prefix/bin"
install -m 755 "$root/target/release/silkai" "$bin"

mkdir -p "$config_dir"
if [[ ! -f "$config_dir/config.toml" ]]; then
  install -m 644 "$root/examples/config.toml" "$config_dir/config.toml"
  echo "Wrote $config_dir/config.toml"
else
  echo "Kept existing $config_dir/config.toml"
fi

if command -v systemctl >/dev/null 2>&1; then
  mkdir -p "$unit_dir"
  sed "s|@BIN@|$bin|g" "$root/contrib/systemd/silkai.service" >"$unit_dir/silkai.service"
  systemctl --user daemon-reload
  echo "systemd unit: $unit_dir/silkai.service"
  echo "Start with:  systemctl --user enable --now silkai"
fi

echo "Installed $bin"
echo "Health check: curl -s http://127.0.0.1:8080/health"
