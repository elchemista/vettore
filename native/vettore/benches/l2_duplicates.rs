//! Run with `cargo bench --bench l2_duplicates --locked`.
//! Measures the production kernel without NIF, allocation, or sorting overhead.

// Cargo's bench build sets cfg(test), but this harness does not run unit tests.
#[allow(dead_code, unused_imports)]
#[path = "../src/distances.rs"]
mod distances;

use std::hint::black_box;
use std::time::Instant;

fn main() {
    const ITERATIONS: usize = 200_000;
    println!("L2 kernel: median of 7 rounds, {ITERATIONS} calls/round (ns/call)");
    for dimensions in [384, 768, 769] {
        let left = (0..dimensions)
            .map(|i| (i % 31) as f32 / 31.0)
            .collect::<Vec<_>>();
        let distinct = left.iter().map(|x| x + 0.125).collect::<Vec<_>>();
        let mut nearly_equal = left.clone();
        nearly_equal[dimensions - 1] += 1.0e-6;
        // Separate allocations: pointer identity must not bypass comparison.
        let identical = left.clone();
        for (label, right) in [
            ("identical", identical),
            ("near duplicate", nearly_equal),
            ("distinct", distinct),
        ] {
            let kernel = black_box(distances::l2 as fn(&[f32], &[f32]) -> f32);
            assert_eq!(kernel(&left, &right) == 0.0, label == "identical");
            for _ in 0..10_000 {
                black_box(kernel(black_box(&left), black_box(&right)));
            }
            let mut samples = [0.0; 7];
            for sample in &mut samples {
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    black_box(kernel(black_box(&left), black_box(&right)));
                }
                *sample = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{dimensions:4}D {label:14}: {:8.1} ns (min {:.1}, max {:.1})",
                samples[3], samples[0], samples[6]
            );
        }
    }
}
