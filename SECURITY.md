# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's private reporting:
[Report a vulnerability](https://github.com/andrecolin/silkai/security/advisories/new).
It reaches the maintainer privately and lets us fix the issue before it is
public. Expect a first reply within about a week.

Tell us what you can: the version or commit, the config that triggers it, and
what an attacker gets. A proof of concept helps but is not required.

## Supported versions

SilkAI is pre-1.0. Fixes land on `main`; there are no backports to older tags.

## Scope

SilkAI is a daemon that owns your GPU and loads models you name in a config
file. Some things are by design, not vulnerabilities:

- **The HTTP and WebSocket listeners have no authentication.** Bind them to
  `127.0.0.1` or put them behind a reverse proxy you control. Exposing the
  daemon to a hostile network is a deployment choice, not a bug in SilkAI.
- **The `process` engine runs the `cmd` from your config.** Anyone who can
  write your config file can run commands as you. Treat the config as trusted.
- **Models you point at are executed as you configure them.** SilkAI does not
  sandbox model weights or engine processes.

In scope: anything that lets a plain HTTP or WebSocket client do more than the
API allows — escape the configured models, read files, run commands, crash the
daemon, or take another tenant's GPU slot.
