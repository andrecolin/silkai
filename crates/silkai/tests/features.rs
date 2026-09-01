const MANIFEST: &str = include_str!("../Cargo.toml");

#[test]
fn llama_gpu_backends_are_cargo_features() {
    for name in ["cuda", "vulkan", "metal"] {
        let needle = format!("{name} = [\"llama\"");
        assert!(
            MANIFEST.contains(&needle),
            "silkai Cargo.toml should declare {name} as a feature that enables llama, missing {needle:?}"
        );
    }
}
