use silkai_sched::{
    Action, GpuBudget, ModelSpec, Priority, Resources, Scheduler, SubmitResult, Tier,
};

fn two_gpu_resources() -> Resources {
    Resources {
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
    }
}

fn writer() -> ModelSpec {
    ModelSpec {
        name: "write".into(),
        vram_gb: 26.0,
        ram_gb: 26.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
    }
}

fn indexer() -> ModelSpec {
    ModelSpec {
        name: "index".into(),
        vram_gb: 10.0,
        ram_gb: 10.0,
        priority: Priority::Background,
        exclusive: false,
        slots: 1,
        keep_warm: true,
        gpu: None,
    }
}

fn started(r: &SubmitResult) -> bool {
    matches!(
        r,
        SubmitResult::Accepted { actions, .. }
            if actions.iter().any(|a| matches!(a, Action::Start { .. }))
    )
}

#[test]
fn eighty_and_thirty_do_not_fit_one_gpu() {
    let mut s = Scheduler::new(
        Resources::single(29.0, 96.0),
        vec![writer(), indexer()],
    )
    .unwrap();
    assert!(started(&s.submit("write")));
    let r = s.submit("index");
    assert!(matches!(r, SubmitResult::Accepted { .. }));
    assert!(!started(&r));
    assert_eq!(s.queued("index"), 1);
}

#[test]
fn eighty_and_thirty_run_on_two_gpus() {
    let mut s = Scheduler::new(two_gpu_resources(), vec![writer(), indexer()]).unwrap();
    assert!(started(&s.submit("write")));
    assert!(started(&s.submit("index")));
    assert_eq!(s.tier("write"), Tier::Bench);
    assert_eq!(s.tier("index"), Tier::Bench);
    assert_eq!(s.gpu_of("write"), Some(0));
    assert_eq!(s.gpu_of("index"), Some(1));
}

#[test]
fn exclusive_on_gpu0_does_not_evict_gpu1() {
    let mut s = Scheduler::new(two_gpu_resources(), vec![writer(), indexer()]).unwrap();
    s.submit("index");
    s.submit("write");
    assert_eq!(s.tier("index"), Tier::Bench);
    assert_eq!(s.tier("write"), Tier::Bench);
    assert_ne!(s.gpu_of("write"), s.gpu_of("index"));
}

#[test]
fn pin_places_on_requested_gpu() {
    let mut index = indexer();
    index.gpu = Some(1);
    let mut s = Scheduler::new(two_gpu_resources(), vec![writer(), index]).unwrap();
    s.submit("index");
    assert_eq!(s.gpu_of("index"), Some(1));
}

#[test]
fn bigger_than_every_gpu_is_too_large() {
    let huge = ModelSpec {
        name: "huge".into(),
        vram_gb: 40.0,
        ram_gb: 40.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
    };
    let mut s = Scheduler::new(two_gpu_resources(), vec![huge]).unwrap();
    assert!(matches!(
        s.submit("huge"),
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}
