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
        gpus: Vec::new(),
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
        gpus: Vec::new(),
    }
}

fn started(r: &SubmitResult) -> bool {
    matches!(
        r,
        SubmitResult::Accepted { actions, .. }
            if actions.iter().any(|a| matches!(a, Action::Start { .. }))
    )
}

fn split_huge() -> ModelSpec {
    ModelSpec {
        name: "huge".into(),
        vram_gb: 40.0,
        ram_gb: 40.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
        gpus: vec![0, 1],
    }
}

fn pin_exclusive(name: &str, gpu: u32) -> ModelSpec {
    ModelSpec {
        name: name.into(),
        vram_gb: 8.0,
        ram_gb: 8.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: Some(gpu),
        gpus: Vec::new(),
    }
}

fn job_id(r: SubmitResult) -> silkai_sched::JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        other => panic!("{other:?}"),
    }
}

#[test]
fn eighty_and_thirty_do_not_fit_one_gpu() {
    let mut s = Scheduler::new(Resources::single(29.0, 96.0), vec![writer(), indexer()]).unwrap();
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
fn load_actions_name_the_gpu() {
    let mut s = Scheduler::new(two_gpu_resources(), vec![writer(), indexer()]).unwrap();
    match s.submit("write") {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions.iter().any(|a| matches!(
                a,
                Action::Load { model, gpu } if model == "write" && *gpu == 0
            )));
        }
        other => panic!("{other:?}"),
    }
    match s.submit("index") {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions.iter().any(|a| matches!(
                a,
                Action::Load { model, gpu } if model == "index" && *gpu == 1
            )));
        }
        other => panic!("{other:?}"),
    }
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
        gpus: Vec::new(),
    };
    let mut s = Scheduler::new(two_gpu_resources(), vec![huge]).unwrap();
    assert!(matches!(
        s.submit("huge"),
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}

#[test]
fn forty_gb_split_runs_on_two_29gb_cards() {
    let mut s = Scheduler::new(two_gpu_resources(), vec![split_huge()]).unwrap();
    assert!(started(&s.submit("huge")));
    assert_eq!(s.tier("huge"), Tier::Bench);
    assert_eq!(s.gpu_of("huge"), Some(0));
    let st = s.status();
    let g0 = st.gpus.iter().find(|g| g.id == 0).unwrap();
    let g1 = st.gpus.iter().find(|g| g.id == 1).unwrap();
    assert_eq!(g0.used_gb, 20.0);
    assert_eq!(g1.used_gb, 20.0);
}

#[test]
fn exclusive_split_blocks_both_cards() {
    let mut small = indexer();
    small.vram_gb = 5.0;
    small.ram_gb = 5.0;
    let mut s = Scheduler::new(two_gpu_resources(), vec![split_huge(), small]).unwrap();
    assert!(started(&s.submit("huge")));
    let r = s.submit("index");
    assert!(!started(&r));
    assert_eq!(s.queued("index"), 1);
    assert_eq!(s.tier("index"), Tier::Cupboard);
}

#[test]
fn missing_listed_gpu_is_too_large() {
    let mut huge = split_huge();
    huge.gpus = vec![0, 99];
    let mut s = Scheduler::new(two_gpu_resources(), vec![huge]).unwrap();
    assert!(matches!(
        s.submit("huge"),
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}

#[test]
fn split_on_single_card_resources_is_too_large() {
    let mut s = Scheduler::new(Resources::single(29.0, 96.0), vec![split_huge()]).unwrap();
    assert!(matches!(
        s.submit("huge"),
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}

#[test]
fn single_id_gpus_list_is_not_a_pin() {
    let mut index = indexer();
    index.gpus = vec![1];
    let mut s = Scheduler::new(two_gpu_resources(), vec![writer(), index]).unwrap();
    s.submit("index");
    assert_eq!(s.gpu_of("index"), Some(0));
}

#[test]
fn split_list_wins_over_gpu_pin() {
    let mut huge = split_huge();
    huge.gpu = Some(1);
    let mut s = Scheduler::new(two_gpu_resources(), vec![huge]).unwrap();
    assert!(started(&s.submit("huge")));
    assert_eq!(s.gpu_of("huge"), Some(0));
    let st = s.status();
    assert_eq!(st.gpus.iter().find(|g| g.id == 0).unwrap().used_gb, 20.0);
    assert_eq!(st.gpus.iter().find(|g| g.id == 1).unwrap().used_gb, 20.0);
}

#[test]
fn split_evicts_idle_copies_on_both_cards() {
    let mut s = Scheduler::new(
        two_gpu_resources(),
        vec![pin_exclusive("a", 0), pin_exclusive("b", 1), split_huge()],
    )
    .unwrap();
    let a = job_id(s.submit("a"));
    let b = job_id(s.submit("b"));
    s.finish(a);
    s.finish(b);
    assert_eq!(s.tier("a"), Tier::Bench);
    assert_eq!(s.tier("b"), Tier::Bench);
    match s.submit("huge") {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Sleep { model } if model == "a")));
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Sleep { model } if model == "b")));
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Start { model, .. } if model == "huge")));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(s.tier("huge"), Tier::Bench);
    assert_eq!(s.tier("a"), Tier::Shelf);
    assert_eq!(s.tier("b"), Tier::Shelf);
}

#[test]
fn exclusive_split_preempts_running_neighbors_on_both_cards() {
    let mut s = Scheduler::new(
        two_gpu_resources(),
        vec![pin_exclusive("a", 0), pin_exclusive("b", 1), split_huge()],
    )
    .unwrap();
    let a = job_id(s.submit("a"));
    let b = job_id(s.submit("b"));
    match s.submit("huge") {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Preempt { job_id } if *job_id == a)));
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Preempt { job_id } if *job_id == b)));
            assert!(actions
                .iter()
                .any(|act| matches!(act, Action::Start { model, .. } if model == "huge")));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(s.tier("huge"), Tier::Bench);
    assert_eq!(s.queued("a"), 1);
    assert_eq!(s.queued("b"), 1);
}

#[test]
fn live_on_one_card_blocks_split_preempt() {
    let mut huge = split_huge();
    huge.priority = Priority::Live;
    let talk = ModelSpec {
        name: "talk".into(),
        vram_gb: 8.0,
        ram_gb: 8.0,
        priority: Priority::Live,
        exclusive: false,
        slots: 1,
        keep_warm: true,
        gpu: Some(1),
        gpus: Vec::new(),
    };
    let mut s = Scheduler::new(
        two_gpu_resources(),
        vec![pin_exclusive("write", 0), talk, huge],
    )
    .unwrap();
    assert!(started(&s.submit("write")));
    assert!(started(&s.submit("talk")));
    let r = s.submit("huge");
    match &r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(!actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
            assert!(!started(&r));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(s.queued("huge"), 1);
    assert_eq!(s.tier("write"), Tier::Bench);
    assert_eq!(s.running("write"), 1);
    assert_eq!(s.tier("talk"), Tier::Bench);
    assert_eq!(s.running("talk"), 1);
}
