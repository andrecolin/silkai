use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{GpuBudget, ModelSpec, Priority, Resources, Scheduler, Tier};

#[test]
fn status_lists_each_model_tier_and_counts() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    s.submit("whisper");
    s.submit("whisper");
    s.submit("whisper"); // queued
    let snap = s.status();
    let w = snap.models.iter().find(|m| m.name == "whisper").unwrap();
    assert_eq!(w.tier, Tier::Bench);
    assert_eq!(w.running, 2);
    assert_eq!(w.queued, 1);
    assert_eq!(snap.gpu_used_gb, 12.0);
    assert_eq!(snap.gpus.len(), 1);
    assert_eq!(snap.gpus[0].id, 0);
    assert_eq!(snap.gpus[0].used_gb, 12.0);
    assert_eq!(snap.gpus[0].schedulable_gb, 29.0);
}

#[test]
fn status_lists_used_per_gpu() {
    let resources = Resources {
        gpu_schedulable_gb: 29.0,
        ram_shelf_gb: 96.0,
        gpus: vec![
            GpuBudget {
                id: 0,
                schedulable_gb: 29.0,
            },
            GpuBudget {
                id: 1,
                schedulable_gb: 29.0,
            },
        ],
    };
    let write = ModelSpec {
        name: "write".into(),
        vram_gb: 26.0,
        ram_gb: 26.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
        gpus: Vec::new(),
    };
    let index = ModelSpec {
        name: "index".into(),
        vram_gb: 10.0,
        ram_gb: 10.0,
        priority: Priority::Background,
        exclusive: false,
        slots: 1,
        keep_warm: true,
        gpu: None,
        gpus: Vec::new(),
    };
    let mut s = Scheduler::new(resources, vec![write, index]).unwrap();
    s.submit("write");
    s.submit("index");
    let snap = s.status();
    let g0 = snap.gpus.iter().find(|g| g.id == 0).unwrap();
    let g1 = snap.gpus.iter().find(|g| g.id == 1).unwrap();
    assert_eq!(g0.used_gb, 26.0);
    assert_eq!(g0.schedulable_gb, 29.0);
    assert_eq!(g1.used_gb, 10.0);
    assert_eq!(g1.schedulable_gb, 29.0);
    assert_eq!(snap.gpu_used_gb, 36.0);
}

#[test]
fn adopt_seeds_residency_and_prefetch_skips_it() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    s.adopt("soap", Tier::Bench, vec![0]);
    s.adopt("whisper", Tier::Shelf, vec![]);
    s.adopt("nobody", Tier::Bench, vec![0]);
    assert_eq!(s.tier("soap"), Tier::Bench);
    assert_eq!(s.gpu_of("soap"), Some(0));
    assert_eq!(s.tier("whisper"), Tier::Shelf);
    assert_eq!(s.gpu_used_gb(), 28.0);
    let warmed: Vec<String> = s
        .prefetch()
        .into_iter()
        .filter_map(|a| match a {
            silkai_sched::Action::Warm { model } => Some(model),
            _ => None,
        })
        .collect();
    // chart-scan is still in the cupboard; the adopted two are not re-warmed.
    assert_eq!(warmed, vec!["chart-scan".to_string()]);
}
