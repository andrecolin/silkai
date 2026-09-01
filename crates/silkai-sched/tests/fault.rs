use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, Scheduler, SubmitResult, Tier};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        other => panic!("expected accepted, got {other:?}"),
    }
}

#[test]
fn fault_marks_model_cupboard_and_next_submit_loads() {
    let mut s = sched();
    let job = job_id(s.submit("soap"));
    assert_eq!(s.tier("soap"), Tier::Bench);
    let actions = s.fault(job);
    assert!(actions
        .iter()
        .any(|a| matches!(a, Action::Discard { model } if model == "soap")));
    assert_eq!(s.tier("soap"), Tier::Cupboard);
    assert_eq!(s.running("soap"), 0);
    assert_eq!(s.gpu_of("soap"), None);
    assert_eq!(s.gpu_used_gb(), 0.0);
    match s.submit("soap") {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Load { model, .. } if model == "soap")));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn fault_queued_job_leaves_running_neighbor() {
    let mut s = sched();
    s.submit("whisper");
    let soap = job_id(s.submit("soap"));
    assert_eq!(s.queued("soap"), 1);
    assert!(s.fault(soap).is_empty());
    assert_eq!(s.queued("soap"), 0);
    assert_eq!(s.running("whisper"), 1);
    assert_eq!(s.tier("whisper"), Tier::Bench);
}
