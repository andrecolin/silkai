use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, Resources, Scheduler, SubmitResult, Tier};

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
fn live_preempts_running_soap_and_requeues_it_at_head() {
    let mut s = sched();
    let soap_id = job_id(s.submit("soap"));
    let r = s.submit("whisper");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Preempt { job_id } if *job_id == soap_id)));
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Sleep { model } if model == "soap")));
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "whisper")));
        }
        _ => panic!("whisper should accept"),
    }
    assert_eq!(s.tier("whisper"), Tier::Bench);
    assert_eq!(s.tier("soap"), Tier::Shelf);
    assert_eq!(s.running("soap"), 0);
    assert_eq!(s.queued("soap"), 1);
    assert_eq!(s.running("whisper"), 1);
}

#[test]
fn soap_does_not_preempt_live_whisper() {
    let mut s = sched();
    s.submit("whisper");
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, job_id: _ } => {
            assert!(!actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
            assert!(!actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "soap")));
        }
        _ => panic!("soap queued"),
    }
    assert_eq!(s.tier("whisper"), Tier::Bench);
    assert_eq!(s.queued("soap"), 1);
}

#[test]
fn background_does_not_preempt_live() {
    // Two whisper jobs still cost 12 GB (slots share one copy). On 21 GB,
    // chart-scan 10 does not fit and must queue rather than preempt live.
    let mut s = Scheduler::new(
        Resources::single(21.0, 96.0),
        clinic_models(),
    )
    .unwrap();
    s.submit("whisper");
    s.submit("whisper");
    let r = s.submit("chart-scan");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(!actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
            assert!(!actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "chart-scan")));
        }
        _ => panic!("accepted queue"),
    }
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn exclusive_preempts_running_background() {
    let mut s = sched();
    let scan = job_id(s.submit("chart-scan"));
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Preempt { job_id } if *job_id == scan)));
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "soap")));
        }
        _ => panic!("soap should start"),
    }
}

#[test]
fn preempted_soap_restarts_after_whisper_finishes() {
    let mut s = sched();
    let soap_id = job_id(s.submit("soap"));
    let w = job_id(s.submit("whisper"));
    let actions = s.finish(w);
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::Start { model, job_id, .. } if model == "soap" && *job_id == soap_id
    )));
}

#[test]
fn queued_exclusive_is_admitted_by_preempting_background_when_live_ends() {
    let mut s = sched();
    let w = job_id(s.submit("whisper"));
    let _scan = job_id(s.submit("chart-scan"));
    let soap_id = job_id(s.submit("soap"));
    assert_eq!(s.queued("soap"), 1);
    let actions = s.finish(w);
    assert!(actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::Start { model, job_id, .. } if model == "soap" && *job_id == soap_id
    )));
}

#[test]
fn later_soap_submit_does_not_jump_queued_soap() {
    let mut s = sched();
    let w = job_id(s.submit("whisper"));
    let _scan = job_id(s.submit("chart-scan"));
    let soap_a = job_id(s.submit("soap"));
    s.finish(w); // should start soap_a by preempting scan
    let soap_b = s.submit("soap");
    match soap_b {
        SubmitResult::Accepted { job_id, actions } => {
            assert_ne!(job_id, soap_a);
            assert!(!actions.iter().any(|a| matches!(a, Action::Start { .. })));
        }
        _ => panic!("queued"),
    }
    assert_eq!(s.running("soap"), 1);
    assert_eq!(s.queued("soap"), 1);
}

#[test]
fn two_queued_soap_notes_keep_resident_copy() {
    let mut s = sched();
    let w = job_id(s.submit("whisper"));
    let _scan = job_id(s.submit("chart-scan"));
    let soap_a = job_id(s.submit("soap"));
    let soap_b = job_id(s.submit("soap"));
    let after_w = s.finish(w);
    assert!(after_w.iter().any(|a| matches!(
        a,
        Action::Start { model, job_id, .. } if model == "soap" && *job_id == soap_a
    )));
    let after_a = s.finish(soap_a);
    assert!(after_a.iter().any(|a| matches!(
        a,
        Action::Start { model, job_id, .. } if model == "soap" && *job_id == soap_b
    )));
    assert!(!after_a.iter().any(|a| matches!(
        a,
        Action::Load { model } | Action::Wake { model } | Action::Sleep { model }
        if model == "soap" || model == "chart-scan"
    )));
    assert!(!after_a
        .iter()
        .any(|a| matches!(a, Action::Start { model, .. } if model == "chart-scan")));
    assert!(!after_a.iter().any(|a| matches!(a, Action::Preempt { .. })));
    assert_eq!(s.tier("soap"), Tier::Bench);
    assert_eq!(s.running("soap"), 1);
}
