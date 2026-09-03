use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, Scheduler, SubmitResult};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn load_count(r: &SubmitResult) -> usize {
    match r {
        SubmitResult::Accepted { actions, .. } => actions
            .iter()
            .filter(|a| matches!(a, Action::Load { .. } | Action::Wake { .. }))
            .count(),
        _ => 0,
    }
}

fn started(r: &SubmitResult) -> bool {
    matches!(r, SubmitResult::Accepted { actions, .. } if actions.iter().any(|a| matches!(a, Action::Start { .. })))
}

#[test]
fn two_whisper_jobs_one_load() {
    let mut s = sched();
    let a = s.submit("whisper");
    assert_eq!(load_count(&a), 1);
    assert!(started(&a));
    let b = s.submit("whisper");
    assert_eq!(load_count(&b), 0);
    assert!(started(&b));
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn third_whisper_queues_at_two_slots() {
    let mut s = sched();
    s.submit("whisper");
    s.submit("whisper");
    let c = s.submit("whisper");
    assert!(!started(&c));
    assert_eq!(s.queued("whisper"), 1);
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn soap_second_job_queues_on_one_slot() {
    let mut s = sched();
    let a = s.submit("soap");
    assert!(started(&a));
    let b = s.submit("soap");
    assert!(!started(&b));
    assert_eq!(s.queued("soap"), 1);
    assert_eq!(s.running("soap"), 1);
    match (&a, &b) {
        (
            SubmitResult::Accepted { job_id: id_a, .. },
            SubmitResult::Accepted { job_id: id_b, .. },
        ) => assert_ne!(id_a, id_b),
        _ => panic!("both accepted"),
    }
}
