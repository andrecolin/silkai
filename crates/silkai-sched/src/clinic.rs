use crate::types::{ModelSpec, Priority, Resources};

pub fn clinic_resources() -> Resources {
    Resources::single(29.0, 96.0)
}

pub fn clinic_models() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            name: "whisper".into(),
            vram_gb: 12.0,
            ram_gb: 12.0,
            priority: Priority::Live,
            exclusive: false,
            slots: 2,
            keep_warm: true,
            gpu: None,
            gpus: Vec::new(),
        },
        ModelSpec {
            name: "soap".into(),
            vram_gb: 28.0,
            ram_gb: 28.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
            gpu: None,
            gpus: Vec::new(),
        },
        ModelSpec {
            name: "chart-scan".into(),
            vram_gb: 10.0,
            ram_gb: 10.0,
            priority: Priority::Background,
            exclusive: false,
            slots: 1,
            keep_warm: true,
            gpu: None,
            gpus: Vec::new(),
        },
    ]
}
