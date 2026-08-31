use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Scheduler, Tier};

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
}
