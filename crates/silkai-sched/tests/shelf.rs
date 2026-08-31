use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, ModelSpec, Priority, Resources, Scheduler, SubmitResult, Tier};

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        _ => panic!("expected accepted"),
    }
}

#[test]
fn prefetch_warms_shelf_not_bench() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    let actions = s.prefetch();
    assert!(actions
        .iter()
        .any(|a| matches!(a, Action::Warm { model } if model == "whisper")));
    assert!(actions
        .iter()
        .any(|a| matches!(a, Action::Warm { model } if model == "soap")));
    assert_eq!(s.tier("whisper"), Tier::Shelf);
    assert_eq!(s.tier("soap"), Tier::Shelf);
    assert_eq!(s.gpu_used_gb(), 0.0);
    assert_eq!(s.ram_used_gb(), 12.0 + 28.0 + 10.0);
}

#[test]
fn second_run_wakes_from_shelf_not_load() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    s.prefetch();
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Wake { model } if model == "soap")));
            assert!(!actions.iter().any(|a| matches!(a, Action::Load { .. })));
        }
        _ => panic!(),
    }
}

#[test]
fn ram_pressure_discards_lru_shelf() {
    let res = Resources::single(29.0, 30.0);
    let mut s = Scheduler::new(res, clinic_models()).unwrap();
    let actions = s.prefetch();
    assert!(s.ram_used_gb() <= 30.0 + 1e-6);
    let cupboard = ["whisper", "soap", "chart-scan"]
        .iter()
        .filter(|n| s.tier(n) == Tier::Cupboard)
        .count();
    assert!(cupboard >= 1);
    let _ = actions;
}

#[test]
fn keep_warm_false_goes_cupboard_on_evict() {
    let models = vec![
        ModelSpec {
            name: "temp".into(),
            vram_gb: 12.0,
            ram_gb: 12.0,
            priority: Priority::Normal,
            exclusive: false,
            slots: 1,
            keep_warm: false,
            gpu: None,
        },
        ModelSpec {
            name: "big".into(),
            vram_gb: 28.0,
            ram_gb: 28.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
            gpu: None,
        },
    ];
    let mut s = Scheduler::new(clinic_resources(), models).unwrap();
    let id = job_id(s.submit("temp"));
    s.finish(id);
    s.submit("big");
    assert_eq!(s.tier("temp"), Tier::Cupboard);
}
