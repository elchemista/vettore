//! Independent numerical oracles and end-to-end native ranking regressions.

use crate::distances::{self, Metric};
use crate::flat::FlatIndex;
use crate::hnsw::{HnswIndex, HnswParams};

fn hnsw(metric: Metric) -> HnswIndex {
    HnswIndex::new(
        metric,
        HnswParams {
            m: 8,
            m0: 16,
            ef_construction: 64,
            ef_search: 64,
            max_level: 8,
        },
    )
    .unwrap()
}

#[test]
fn dot_recovers_underflowed_products_before_rounding_the_sum() {
    for dimensions in [1, 7, 8, 9, 383, 768, 769, 787, 1024, 1536, 3072] {
        for scale in [1.0e-24_f32, 1.0e-23, 1.0e-22, 1.0e-20, 1.0e-19] {
            for sign in [-1.0_f32, 1.0] {
                let left = vec![scale; dimensions];
                let right = vec![scale * sign; dimensions];
                let expected =
                    (f64::from(scale).powi(2) * f64::from(sign) * dimensions as f64) as f32;
                for (metric, value) in [
                    (Metric::InnerProduct, expected),
                    (Metric::NegativeInnerProduct, -expected),
                ] {
                    assert_eq!(
                        distances::compute(metric, &left, &right),
                        Ok(value),
                        "{metric:?}, dimensions={dimensions}, scale={scale}, sign={sign}"
                    );
                }
            }
        }
    }
    let left = vec![1.0e-23; 768];
    let cancelling = (0..768)
        .map(|i| if i % 2 == 0 { 1.0e-23 } else { -1.0e-23 })
        .collect::<Vec<_>>();
    assert_eq!(distances::dot(&left, &cancelling), 0.0);
    assert!(distances::dot(&left, &left) > 0.0);
}

#[test]
fn tiny_dot_ranking_survives_in_flat_and_hnsw() {
    for metric in [Metric::InnerProduct, Metric::NegativeInnerProduct] {
        let mut flat = FlatIndex::new(metric);
        let mut graph = hnsw(metric);
        for (id, multiplier) in [
            ("a-negative", -2.0),
            ("b-zero", 0.0),
            ("m-near", 1.0),
            ("z-best", 2.0),
        ] {
            let vector = vec![1.0e-23 * multiplier; 768];
            flat.insert(id.into(), vector.clone()).unwrap();
            graph.insert(id.into(), vector).unwrap();
        }
        for hits in [
            flat.search(&[1.0e-23; 768], 4).unwrap(),
            graph.search(&[1.0e-23; 768], 4).unwrap(),
        ] {
            assert_eq!(
                hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
                ["z-best", "m-near", "b-zero", "a-negative"]
            );
        }
    }
}

#[test]
fn final_signed_zero_scores_tie_by_id_in_every_native_ranker() {
    assert_eq!(
        distances::dot(&[1.0e-30], &[-1.0e-30]).to_bits(),
        (-0.0f32).to_bits()
    );
    let rows = vec![
        ("a-negative".to_string(), vec![-1.0e-30]),
        ("z-positive".to_string(), vec![1.0e-30]),
        ("m-zero".to_string(), vec![0.0]),
    ];
    for metric in [Metric::InnerProduct, Metric::NegativeInnerProduct] {
        let mut flat = FlatIndex::new(metric);
        let mut graph = hnsw(metric);
        flat.insert_many(rows.clone()).unwrap();
        graph.insert_many(rows.clone()).unwrap();
        for hits in [
            flat.search(&[1.0e-30], 3).unwrap(),
            graph.search(&[1.0e-30], 3).unwrap(),
            crate::search::vector_top_k(rows.clone(), &[1.0e-30], metric, 1, 3).unwrap(),
        ] {
            assert_eq!(
                hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
                ["a-negative", "m-zero", "z-positive"]
            );
            assert!(hits.iter().all(|(_, raw)| *raw == 0.0));
        }
    }
}

#[test]
fn gpu_dot_preserves_representable_subnormal_results() {
    if !crate::gpu::detected() {
        assert_ne!(std::env::var("VETTORE_REQUIRE_GPU").as_deref(), Ok("1"));
        return;
    }
    for dimensions in [1, 7, 9, 383, 768, 769] {
        for scale in [1.0e-23, 1.0e-22, 1.0e-20] {
            let left = vec![scale; dimensions];
            for sign in [-1.0, 1.0] {
                let right = vec![scale * sign; dimensions];
                for metric in [Metric::InnerProduct, Metric::NegativeInnerProduct] {
                    check_oracle(
                        crate::gpu::metric(&left, &right, metric).unwrap(),
                        metric,
                        &left,
                        &right,
                    );
                }
            }
        }
    }
}

#[test]
fn l2_preserves_tiny_distances_before_rounding_the_final_result() {
    for dimensions in [1, 7, 8, 9, 383, 768, 769, 787, 1024, 1536, 3072] {
        for value in [
            f32::from_bits(1),
            1.0e-30,
            1.0e-23,
            1.0e-22,
            1.0e-20,
            1.0e-19,
        ] {
            let left = vec![value; dimensions];
            let right = vec![0.0; dimensions];
            let squared = f64::from(value).powi(2) * dimensions as f64;
            for (metric, expected) in [
                (Metric::L2, squared.sqrt() as f32),
                (Metric::L2Squared, squared as f32),
            ] {
                let actual = distances::compute(metric, &left, &right).unwrap();
                // Absolute epsilons such as 1e-6 would hide this regression entirely.
                assert_eq!(
                    actual, expected,
                    "{metric:?}, dimensions={dimensions}, value={value}"
                );
                assert_eq!(distances::compute(metric, &left, &left), Ok(0.0));
            }
        }
    }
    // Numeric equality need not imply identical bits; the fallback handles it.
    for metric in [Metric::L2, Metric::L2Squared] {
        assert_eq!(distances::compute(metric, &[0.0; 9], &[-0.0; 9]), Ok(0.0));
    }
}

#[test]
fn tiny_l2_ranks_by_distance_in_flat_and_hnsw() {
    for (metric, scale) in [(Metric::L2, 1.0e-30), (Metric::L2Squared, 1.0e-22)] {
        let mut flat = FlatIndex::new(metric);
        let mut graph = hnsw(metric);
        for (id, multiplier) in [("a-far", 3.0), ("z-near", 1.0), ("m-middle", 2.0)] {
            let vector = vec![scale * multiplier; 9];
            flat.insert(id.into(), vector.clone()).unwrap();
            graph.insert(id.into(), vector).unwrap();
        }
        for hits in [
            flat.search(&[0.0; 9], 3).unwrap(),
            graph.search(&[0.0; 9], 3).unwrap(),
        ] {
            assert_eq!(
                hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
                ["z-near", "m-middle", "a-far"]
            );
            assert!(hits[0].1 > 0.0 && hits[0].1 < hits[1].1 && hits[1].1 < hits[2].1);
        }
    }
}

#[test]
fn cosine_ranking_preserves_distinct_small_similarities() {
    // Near zero, 1 - similarity rounds distinct raw f32 scores to 1.0.
    let mut flat = FlatIndex::new(Metric::Cosine);
    let mut graph = hnsw(Metric::Cosine);
    for (id, x) in [("a-worse", 1.0e-8), ("z-better", 2.0e-8)] {
        flat.insert(id.into(), vec![x, 1.0]).unwrap();
        graph.insert(id.into(), vec![x, 1.0]).unwrap();
    }
    for hits in [
        flat.search(&[1.0, 0.0], 2).unwrap(),
        graph.search(&[1.0, 0.0], 2).unwrap(),
    ] {
        assert_eq!(hits[0].0, "z-better");
        assert!(hits[0].1 > hits[1].1);
    }
    // Near one these decimal examples already coincide at the input f32 boundary.
    assert_eq!(0.99999994_f64 as f32, 0.99999997_f64 as f32);
    let below_one = f32::from_bits(1.0f32.to_bits() - 1);
    assert!(
        distances::rank_value(Metric::Cosine, 1.0)
            < distances::rank_value(Metric::Cosine, below_one)
    );
}

#[test]
fn native_index_calls_never_wait_for_locks_on_normal_schedulers() {
    let mut workload_found = false;
    for nif in rustler::codegen_runtime::inventory::iter::<rustler::Nif> {
        // Rustler's macro registers a static, NUL-terminated name for each NIF.
        let name = unsafe { std::ffi::CStr::from_ptr(nif.name) }
            .to_str()
            .unwrap();
        if name.starts_with("flat_") || name.starts_with("hnsw_") {
            assert_ne!(nif.flags, 0, "{name} may block on a native index lock");
        }
        workload_found |= name == "flat_workload";
    }
    assert!(workload_found);
}

fn reference(metric: Metric, left: &[f32], right: &[f32]) -> (f64, f64) {
    let pairs = || {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| (f64::from(a), f64::from(b)))
    };
    let squared = || pairs().map(|(a, b)| (a - b).powi(2)).sum::<f64>();
    let dot = || pairs().map(|(a, b)| a * b).sum::<f64>();
    let absolute_products = || pairs().map(|(a, b)| (a * b).abs()).sum::<f64>();
    let value = match metric {
        Metric::L2 => squared().sqrt(),
        Metric::L2Squared => squared(),
        Metric::InnerProduct => dot(),
        Metric::NegativeInnerProduct => -dot(),
        Metric::Cosine => {
            let norm = (pairs().map(|(a, _)| a * a).sum::<f64>()
                * pairs().map(|(_, b)| b * b).sum::<f64>())
            .sqrt();
            if norm == 0.0 {
                0.0
            } else {
                (dot() / norm).clamp(-1.0, 1.0)
            }
        }
        Metric::Manhattan => pairs().map(|(a, b)| (a - b).abs()).sum(),
        Metric::Chebyshev => pairs().map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
        Metric::Hamming => pairs().filter(|(a, b)| (*a != 0.0) != (*b != 0.0)).count() as f64,
        Metric::Jaccard => {
            let union = pairs().filter(|(a, b)| *a != 0.0 || *b != 0.0).count();
            let intersection = pairs().filter(|(a, b)| *a != 0.0 && *b != 0.0).count();
            if union == 0 {
                0.0
            } else {
                1.0 - intersection as f64 / union as f64
            }
        }
    };
    // Near cancellation, relative error to the final result alone is undefined.
    let scale = match metric {
        Metric::InnerProduct | Metric::NegativeInnerProduct => absolute_products(),
        Metric::Cosine => 1.0,
        _ => value.abs(),
    };
    (value, scale)
}

fn check_oracle(actual: f32, metric: Metric, left: &[f32], right: &[f32]) {
    let (expected, scale) = reference(metric, left, right);
    let tolerance = scale * 2.0e-6 + f64::from(f32::from_bits(1));
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance,
        "{metric:?}, dimensions={}: expected {expected}, got {actual}, tolerance {tolerance}",
        left.len()
    );
}

#[test]
fn every_cpu_and_gpu_metric_matches_f64_on_seeded_vectors_and_simd_tails() {
    let gpu = crate::gpu::detected();
    assert!(gpu || std::env::var("VETTORE_REQUIRE_GPU").as_deref() != Ok("1"));
    let mut state = 0x12ab_34cd_u32;
    for dimensions in [1, 7, 9, 383, 769, 787, 1024, 1536, 3072] {
        for scale in [1.0e-23, 1.0e-20, 1.0e-10, 1.0, 1.0e10] {
            for _case in 0..4 {
                let mut random = || {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state % 257) as f32 - 128.0) / 128.0 * scale
                };
                let left = (0..dimensions).map(|_| random()).collect::<Vec<_>>();
                let right = (0..dimensions).map(|_| random()).collect::<Vec<_>>();
                for code in 0..9 {
                    let metric = Metric::from_code(code).unwrap();
                    check_oracle(
                        distances::compute(metric, &left, &right).unwrap(),
                        metric,
                        &left,
                        &right,
                    );
                    if gpu {
                        check_oracle(
                            crate::gpu::metric(&left, &right, metric).unwrap(),
                            metric,
                            &left,
                            &right,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn gpu_primitives_preserve_tiny_l2_and_final_subnormal_squared_distances() {
    if !crate::gpu::detected() {
        assert_ne!(std::env::var("VETTORE_REQUIRE_GPU").as_deref(), Ok("1"));
        return;
    }
    for dimensions in [1, 7, 9, 383, 769] {
        for scale in [1.0e-30, 1.0e-22, 1.0e-20] {
            let values = vec![scale; dimensions];
            let zero = vec![0.0; dimensions];
            for metric in [Metric::L2, Metric::L2Squared] {
                check_oracle(
                    crate::gpu::metric(&values, &zero, metric).unwrap(),
                    metric,
                    &values,
                    &zero,
                );
            }
        }
    }
}

#[test]
fn resident_gpu_preserves_small_cosine_ranks_and_sparse_jaccard_differences() {
    if !crate::gpu::detected() {
        assert_ne!(std::env::var("VETTORE_REQUIRE_GPU").as_deref(), Ok("1"));
        return;
    }
    let cosine = crate::flat::FlatResource::new(Metric::Cosine);
    cosine.insert("a-worse".into(), vec![1.0e-8, 1.0]).unwrap();
    cosine.insert("z-better".into(), vec![2.0e-8, 1.0]).unwrap();
    let hits = cosine.search_gpu(&[1.0, 0.0], 2).unwrap();
    assert_eq!(hits[0].0, "z-better");
    assert!(hits[0].1 > hits[1].1);

    let jaccard = crate::flat::FlatResource::new(Metric::Jaccard);
    let query = vec![1.0; 769];
    let mut row = query.clone();
    row[0] = 0.0;
    jaccard
        .insert("one-difference".into(), row.clone())
        .unwrap();
    let hits = jaccard.search_gpu(&query, 1).unwrap();
    check_oracle(hits[0].1, Metric::Jaccard, &query, &row);
}
