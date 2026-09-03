//! Prometheus text exposition, written by hand: the format is a few lines
//! and a crate would be the only new dependency in the daemon.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Mutex;

use crate::status::Status;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Counters {
    pub loads: u64,
    pub wakes: u64,
    pub sleeps: u64,
    pub preempts: u64,
    pub faults: u64,
    /// Seconds spent in load and wake calls.
    pub load_secs: f64,
}

#[derive(Default)]
pub struct Metrics {
    per_model: Mutex<HashMap<String, Counters>>,
}

impl Metrics {
    pub fn bump(&self, model: &str, f: impl FnOnce(&mut Counters)) {
        let mut map = self.per_model.lock().expect("metrics mutex");
        f(map.entry(model.to_string()).or_default());
    }

    pub fn snapshot(&self) -> HashMap<String, Counters> {
        self.per_model.lock().expect("metrics mutex").clone()
    }
}

const STATES: [&str; 5] = ["cupboard", "shelf", "loading", "bench", "sleeping"];

pub fn render(status: &Status, counters: &HashMap<String, Counters>) -> String {
    let mut out = String::new();
    gauge(
        &mut out,
        "silkai_gpu_schedulable_gb",
        "GB the scheduler may use per card",
    );
    for g in &status.gpus {
        line(
            &mut out,
            "silkai_gpu_schedulable_gb",
            &[("gpu", &g.id.to_string())],
            g.schedulable_gb,
        );
    }
    gauge(
        &mut out,
        "silkai_gpu_budget_used_gb",
        "Sum of vram_gb for models on the card",
    );
    for g in &status.gpus {
        line(
            &mut out,
            "silkai_gpu_budget_used_gb",
            &[("gpu", &g.id.to_string())],
            g.used_gb,
        );
    }
    gauge(
        &mut out,
        "silkai_gpu_measured_used_gb",
        "GB in use as reported by the driver",
    );
    for g in &status.gpus {
        if let Some(m) = g.measured_used_gb {
            line(
                &mut out,
                "silkai_gpu_measured_used_gb",
                &[("gpu", &g.id.to_string())],
                m,
            );
        }
    }
    gauge(
        &mut out,
        "silkai_ram_used_gb",
        "GB of warm copies held in host RAM",
    );
    line(&mut out, "silkai_ram_used_gb", &[], status.ram_used_gb);

    gauge(
        &mut out,
        "silkai_model_state",
        "1 for the model's current state",
    );
    for m in &status.models {
        for s in STATES {
            let v = if m.state == s { 1.0 } else { 0.0 };
            line(
                &mut out,
                "silkai_model_state",
                &[("model", &m.name), ("state", s)],
                v,
            );
        }
    }
    per_model_gauge(
        &mut out,
        status,
        "silkai_model_running",
        "Jobs generating now",
        |m| m.running as f64,
    );
    per_model_gauge(
        &mut out,
        status,
        "silkai_model_queued",
        "Jobs waiting for the card",
        |m| m.queued as f64,
    );
    per_model_gauge(
        &mut out,
        status,
        "silkai_model_sessions",
        "Open session sockets",
        |m| m.sessions as f64,
    );
    per_model_gauge(
        &mut out,
        status,
        "silkai_model_budget_gb",
        "Configured vram_gb",
        |m| m.budget_gb,
    );
    gauge(
        &mut out,
        "silkai_model_measured_gb",
        "GB the driver attributes to the model's process",
    );
    for m in &status.models {
        if let Some(v) = m.measured_gb {
            line(
                &mut out,
                "silkai_model_measured_gb",
                &[("model", &m.name)],
                v,
            );
        }
    }

    counter(
        &mut out,
        "silkai_loads_total",
        "Loads from disk",
        counters,
        |c| c.loads as f64,
    );
    counter(
        &mut out,
        "silkai_wakes_total",
        "Wakes from the shelf",
        counters,
        |c| c.wakes as f64,
    );
    counter(
        &mut out,
        "silkai_sleeps_total",
        "Moves to the shelf",
        counters,
        |c| c.sleeps as f64,
    );
    counter(
        &mut out,
        "silkai_preempts_total",
        "Jobs interrupted for a live one",
        counters,
        |c| c.preempts as f64,
    );
    counter(
        &mut out,
        "silkai_faults_total",
        "Engine failures",
        counters,
        |c| c.faults as f64,
    );
    counter(
        &mut out,
        "silkai_load_seconds_sum",
        "Seconds spent loading or waking",
        counters,
        |c| c.load_secs,
    );
    out
}

fn gauge(out: &mut String, name: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge");
}

fn per_model_gauge(
    out: &mut String,
    status: &Status,
    name: &str,
    help: &str,
    value: impl Fn(&crate::status::ModelStatus) -> f64,
) {
    gauge(out, name, help);
    for m in &status.models {
        line(out, name, &[("model", &m.name)], value(m));
    }
}

fn counter(
    out: &mut String,
    name: &str,
    help: &str,
    counters: &HashMap<String, Counters>,
    value: impl Fn(&Counters) -> f64,
) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter");
    let mut names: Vec<&String> = counters.keys().collect();
    names.sort();
    for model in names {
        line(out, name, &[("model", model)], value(&counters[model]));
    }
}

fn line(out: &mut String, name: &str, labels: &[(&str, &str)], value: f64) {
    out.push_str(name);
    if !labels.is_empty() {
        out.push('{');
        for (i, (k, v)) in labels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{k}=\"{}\"", escape(v));
        }
        out.push('}');
    }
    let _ = writeln!(out, " {value}");
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
