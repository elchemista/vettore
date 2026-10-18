//! Compare the production dot kernel with its raw f32 reduction.
//! The raw reduction is a timing baseline, not a numerically safe replacement.

// Cargo's bench build sets cfg(test), but this harness does not run unit tests.
#[allow(dead_code, unused_imports)]
#[path = "../src/distances.rs"]
mod distances;

use std::hint::black_box;
use std::time::Instant;

fn measure(kernel: fn(&[f32], &[f32]) -> f32, left: &[f32], right: &[f32]) -> f64 {
    const ITERATIONS: usize = 100_000;
    let kernel = black_box(kernel);
    for _ in 0..10_000 {
        black_box(kernel(black_box(left), black_box(right)));
    }
    let mut samples = [0.0; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(kernel(black_box(left), black_box(right)));
        }
        *sample = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}

fn main() {
    println!("Dot: median of 7 rounds, 100000 calls/round (ns/call)");
    println!("Raw f32 is only a baseline: it loses representable tiny products.");
    for dimensions in [384, 768, 769, 1536] {
        let mut one_hot_left = vec![0.0; dimensions];
        one_hot_left[dimensions - 1] = 1.0;
        let mut one_hot_right = vec![0.0; dimensions];
        one_hot_right[0] = 1.0;
        let sparse_left = (0..dimensions)
            .map(|i| if i % 8 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let sparse_right = (0..dimensions)
            .map(|i| if i % 8 == 1 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let cancellation = (0..dimensions)
            .map(|i| {
                if dimensions % 2 == 1 && i == dimensions - 1 {
                    0.0
                } else if i % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>();
        for (label, left, right) in [
            ("one-hot orthogonal", one_hot_left, one_hot_right),
            ("sparse orthogonal", sparse_left, sparse_right),
            ("dense cancellation", vec![1.0; dimensions], cancellation),
            (
                "dense nonzero",
                vec![1.0; dimensions],
                vec![0.5; dimensions],
            ),
            (
                "tiny nonzero",
                vec![1.0e-23; dimensions],
                vec![1.0e-23; dimensions],
            ),
        ] {
            let reference = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32;
            assert_eq!(distances::dot(&left, &right), reference);
            if label == "tiny nonzero" {
                assert!(reference > 0.0);
                assert_eq!(distances::simd_dot(&left, &right), 0.0);
            }
            let raw = measure(distances::simd_dot, &left, &right);
            let recovered = measure(distances::dot, &left, &right);
            println!(
                "{dimensions:4}D {label:19}: raw {raw:8.1}, recovered {recovered:8.1}, ratio {:.2}x",
                recovered / raw
            );
        }
    }
}
