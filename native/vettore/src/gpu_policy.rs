//! Deterministic GPU selection and failure policy, independent of driver I/O.

use std::panic::{catch_unwind, AssertUnwindSafe};

pub(crate) fn initialize_first<A, T>(
    mut adapters: Vec<(wgpu::AdapterInfo, A)>,
    allow_software: bool,
    mut initialize: impl FnMut(A) -> Result<T, String>,
) -> Result<T, String> {
    adapters.retain(|(info, _)| {
        info.backend != wgpu::Backend::Noop
            && (allow_software || info.device_type != wgpu::DeviceType::Cpu)
    });
    // Prefer native compute backends over GL for the same device class; then
    // use stable hardware identifiers rather than enumeration order.
    adapters.sort_by(|(left, _), (right, _)| adapter_key(left).cmp(&adapter_key(right)));
    let mut errors = Vec::new();
    for (info, adapter) in adapters {
        match catch_unwind(AssertUnwindSafe(|| initialize(adapter))) {
            Ok(Ok(runtime)) => return Ok(runtime),
            Ok(Err(error)) => errors.push(format!(
                "{} ({}): {error}",
                info.name,
                info.backend.to_str()
            )),
            Err(_) => errors.push(format!(
                "{} ({}): initialization panicked",
                info.name,
                info.backend.to_str()
            )),
        }
    }
    if errors.is_empty() {
        Err("gpu not detected".into())
    } else {
        Err(format!(
            "gpu device initialization failed: {}",
            errors.join("; ")
        ))
    }
}

fn adapter_key(info: &wgpu::AdapterInfo) -> (u8, u8, u32, u32, &str, &str) {
    let device = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let backend = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
        wgpu::Backend::Gl => 1,
        _ => 2,
    };
    (
        device,
        backend,
        info.vendor,
        info.device,
        &info.name,
        info.backend.to_str(),
    )
}

pub(crate) fn runtime_error_requires_reinitialization(error: &str) -> bool {
    error.starts_with("gpu synchronization")
        || error.starts_with("gpu mapping")
        || error.starts_with("gpu mapped range")
        || error.starts_with("gpu readback")
        || error == "gpu operation panicked"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(
        name: &str,
        device_type: wgpu::DeviceType,
        backend: wgpu::Backend,
    ) -> (wgpu::AdapterInfo, String) {
        (
            wgpu::AdapterInfo {
                name: name.into(),
                vendor: 1,
                device: 1,
                device_type,
                driver: String::new(),
                driver_info: String::new(),
                backend,
                device_pci_bus_id: String::new(),
                subgroup_min_size: 4,
                subgroup_max_size: 32,
                transient_saves_memory: None,
                limit_bucket: None,
            },
            name.into(),
        )
    }

    #[test]
    fn tries_other_backends_and_devices_after_errors_or_panics() {
        use wgpu::{Backend::*, DeviceType::*};
        let mut tried = Vec::new();
        let result = initialize_first(
            vec![
                adapter("integrated", IntegratedGpu, Vulkan),
                adapter("discrete-gl", DiscreteGpu, Gl),
                adapter("discrete-vulkan", DiscreteGpu, Vulkan),
            ],
            false,
            |name| {
                tried.push(name.clone());
                match name.as_str() {
                    "discrete-vulkan" => Err("device unavailable".into()),
                    "discrete-gl" => panic!("pipeline validation failed"),
                    _ => Ok(name),
                }
            },
        );
        assert_eq!(result, Ok("integrated".into()));
        assert_eq!(tried, ["discrete-vulkan", "discrete-gl", "integrated"]);
    }

    #[test]
    fn selection_is_stable_filters_software_and_reports_all_failures() {
        use wgpu::{Backend::*, DeviceType::*};
        let devices = vec![
            adapter("b", DiscreteGpu, Vulkan),
            adapter("a", DiscreteGpu, Vulkan),
        ];
        assert_eq!(initialize_first(devices.clone(), false, Ok), Ok("a".into()));
        assert_eq!(
            initialize_first(devices.into_iter().rev().collect(), false, Ok),
            Ok("a".into())
        );
        assert_eq!(
            initialize_first(
                vec![adapter("cpu", Cpu, Vulkan), adapter("noop", Other, Noop)],
                false,
                Ok
            ),
            Err("gpu not detected".into())
        );
        assert_eq!(
            initialize_first(vec![adapter("noop", Other, Noop)], true, Ok),
            Err("gpu not detected".into())
        );
        assert_eq!(
            initialize_first(vec![adapter("cpu", Cpu, Vulkan)], true, Ok),
            Ok("cpu".into())
        );
        assert_eq!(
            initialize_first(
                vec![
                    adapter("virtual", VirtualGpu, Vulkan),
                    adapter("other", Other, Gl)
                ],
                false,
                Ok
            ),
            Ok("virtual".into())
        );
        assert_eq!(
            initialize_first(
                vec![
                    adapter("cpu", Cpu, Vulkan),
                    adapter("other", Other, BrowserWebGpu)
                ],
                true,
                Ok
            ),
            Ok("other".into())
        );
        let error = initialize_first(
            vec![
                adapter("a", DiscreteGpu, Vulkan),
                adapter("b", IntegratedGpu, Gl),
            ],
            false,
            |_| Err::<(), _>("broken".into()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "gpu device initialization failed: a (vulkan): broken; b (gl): broken"
        );
    }

    #[test]
    fn only_runtime_failures_require_a_new_device() {
        for error in [
            "gpu synchronization failed: timeout",
            "gpu mapping callback failed",
            "gpu mapped range failed",
            "gpu readback failed",
            "gpu operation panicked",
        ] {
            assert!(runtime_error_requires_reinitialization(error));
        }
        for error in [
            "gpu numeric range unsupported",
            "gpu workload too large",
            "dimension mismatch",
        ] {
            assert!(!runtime_error_requires_reinitialization(error));
        }
    }
}
