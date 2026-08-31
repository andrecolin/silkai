use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, Scheduler, SubmitResult};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        _ => panic!("expected accepted"),
    }
}

#[test]
fn finishing_soap_starts_queued_soap_without_second_load() {
    let mut s = sched();
    let first = job_id(s.submit("soap"));
    s.submit("soap");
    let actions = s.finish(first);
    assert!(actions.iter().any(|a| matches!(a, Action::Start { .. })));
    assert!(!actions
        .iter()
        .any(|a| matches!(a, Action::Load { .. } | Action::Wake { .. })));
    assert_eq!(s.running("soap"), 1);
    assert_eq!(s.queued("soap"), 0);
}

#[test]
fn finishing_whisper_and_scan_allows_queued_soap() {
    let mut s = sched();
    let w = job_id(s.submit("whisper"));
    let c = job_id(s.submit("chart-scan"));
    s.submit("soap");
    let actions = s.finish(w);
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::Start { model, .. } if model == "soap"
    )));
    assert_eq!(s.queued("soap"), 0);
    assert_eq!(s.tier("soap"), silkai_sched::Tier::Bench);
    assert!(s.finish(c).is_empty());
}

#[test]
fn drop_queued_job_leaves_running_untouched() {
    let mut s = sched();
    s.submit("whisper");
    let soap = job_id(s.submit("soap"));
    assert_eq!(s.queued("soap"), 1);
    assert!(s.drop_job(soap).is_empty());
    assert_eq!(s.queued("soap"), 0);
    assert_eq!(s.running("whisper"), 1);
    assert!(s.drop_job(JobId(99)).is_empty());
}
