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
        Resources {
            gpu_schedulable_gb: 21.0,
            ram_shelf_gb: 96.0,
        },
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
