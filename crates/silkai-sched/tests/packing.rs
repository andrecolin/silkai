use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, Scheduler, SubmitResult, Tier};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn started(result: &SubmitResult) -> bool {
    match result {
        SubmitResult::Accepted { actions, .. } => {
            actions.iter().any(|a| matches!(a, Action::Start { .. }))
        }
        _ => false,
    }
}

#[test]
fn whisper_then_chart_scan_both_run() {
    let mut s = sched();
    let a = s.submit("whisper");
    assert!(started(&a));
    assert_eq!(s.tier("whisper"), Tier::Bench);
    let b = s.submit("chart-scan");
    assert!(started(&b));
    assert_eq!(s.tier("chart-scan"), Tier::Bench);
    assert_eq!(s.gpu_used_gb(), 22.0);
}

#[test]
fn soap_does_not_start_beside_whisper_and_scan() {
    let mut s = sched();
    s.submit("whisper");
    s.submit("chart-scan");
    let r = s.submit("soap");
    assert!(matches!(r, SubmitResult::Accepted { .. }));
    assert!(!started(&r));
    assert_eq!(s.tier("soap"), Tier::Cupboard);
    assert_eq!(s.queued("soap"), 1);
}

#[test]
fn unknown_model_is_rejected() {
    let mut s = sched();
    let r = s.submit("nope");
    assert!(matches!(
        r,
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::UnknownModel
        }
    ));
}

#[test]
fn model_bigger_than_gpu_is_rejected() {
    let mut s = Scheduler::new(
        clinic_resources(),
        vec![silkai_sched::ModelSpec {
            name: "huge".into(),
            vram_gb: 40.0,
            ram_gb: 40.0,
            priority: silkai_sched::Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
            gpu: None,
            gpus: Vec::new(),
        }],
    )
    .unwrap();
    let r = s.submit("huge");
    assert!(matches!(
        r,
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}
