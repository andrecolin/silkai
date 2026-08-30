use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Priority, Resources};

#[test]
fn clinic_budget_is_29_gb_gpu_and_96_gb_shelf() {
    let r: Resources = clinic_resources();
    assert_eq!(r.gpu_schedulable_gb, 29.0);
    assert_eq!(r.ram_shelf_gb, 96.0);
}

#[test]
fn clinic_models_match_spec() {
    let models = clinic_models();
    assert_eq!(models.len(), 3);
    let w = models.iter().find(|m| m.name == "whisper").unwrap();
    assert_eq!(w.vram_gb, 12.0);
    assert_eq!(w.priority, Priority::Live);
    assert!(!w.exclusive);
    assert_eq!(w.slots, 2);
    assert!(w.keep_warm);
    let s = models.iter().find(|m| m.name == "soap").unwrap();
    assert_eq!(s.vram_gb, 28.0);
    assert_eq!(s.priority, Priority::Normal);
    assert!(s.exclusive);
    assert_eq!(s.slots, 1);
    let c = models.iter().find(|m| m.name == "chart-scan").unwrap();
    assert_eq!(c.vram_gb, 10.0);
    assert_eq!(c.priority, Priority::Background);
}

#[test]
fn live_outranks_normal_outranks_background() {
    assert!(Priority::Background < Priority::Normal);
    assert!(Priority::Normal < Priority::Live);
}
