//! Pure validation and reduction helpers shared by the wgpu runtime.

use std::borrow::Cow;

use crate::distances::{self, Metric};

pub(crate) struct PreparedMetric<'a> {
    pub(crate) left: Cow<'a, [f32]>,
    pub(crate) right: Cow<'a, [f32]>,
    result_scale: f64,
}

pub(crate) struct PreparedNormalization {
    pub(crate) vector: Vec<f32>,
    pub(crate) first: f32,
    pub(crate) second: f32,
}

pub(crate) fn metric_code(metric: Metric) -> u32 {
    match metric {
        Metric::L2 => 0,
        Metric::L2Squared => 1,
        Metric::Cosine => 2,
        Metric::InnerProduct => 3,
        Metric::NegativeInnerProduct => 4,
        Metric::Manhattan => 5,
        Metric::Chebyshev => 6,
        Metric::Hamming => 7,
        Metric::Jaccard => 8,
    }
}

pub(crate) fn finish_metric(metric: Metric, partials: &[[f32; 4]]) -> Result<f32, String> {
    if partials
        .iter()
        .flat_map(|partial| partial.iter())
        .any(|value| !value.is_finite())
    {
        return Err(distances::METRIC_OVERFLOW.to_string());
    }

    let sums = partials.iter().fold([0.0f64; 4], |mut sums, partial| {
        for (sum, value) in sums.iter_mut().zip(partial) {
            *sum += f64::from(*value);
        }
        sums
    });

    let value = match metric {
        Metric::L2 => sums[0].sqrt(),
        Metric::L2Squared => sums[0],
        Metric::Cosine => {
            let denominator = (sums[1] * sums[2]).sqrt();
            if denominator == 0.0 {
                0.0
            } else {
                (sums[0] / denominator).clamp(-1.0, 1.0)
            }
        }
        Metric::InnerProduct => sums[0],
        Metric::NegativeInnerProduct => -sums[0],
        Metric::Manhattan | Metric::Hamming => sums[0],
        Metric::Chebyshev => partials
            .iter()
            .map(|partial| f64::from(partial[0]))
            .fold(0.0, f64::max),
        Metric::Jaccard => {
            if sums[1] == 0.0 {
                0.0
            } else {
                1.0 - sums[0] / sums[1]
            }
        }
    };

    if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
        Ok(value as f32)
    } else {
        Err(distances::METRIC_OVERFLOW.to_string())
    }
}

pub(crate) fn prepare_metric<'a>(
    left: &'a [f32],
    right: &'a [f32],
    metric: Metric,
) -> PreparedMetric<'a> {
    match metric {
        Metric::L2 | Metric::L2Squared | Metric::Manhattan | Metric::Chebyshev => {
            let scale = max_abs(left).max(max_abs(right));
            if scale == 0.0 {
                borrowed_metric(left, right)
            } else {
                let result_scale = match metric {
                    Metric::L2Squared => scale * scale,
                    _ => scale,
                };

                PreparedMetric {
                    left: Cow::Owned(scale_vector(left, scale)),
                    right: Cow::Owned(scale_vector(right, scale)),
                    result_scale,
                }
            }
        }
        Metric::Cosine => PreparedMetric {
            left: scaled_or_borrowed(left),
            right: scaled_or_borrowed(right),
            result_scale: 1.0,
        },
        Metric::InnerProduct | Metric::NegativeInnerProduct => {
            let left_scale = max_abs(left);
            let right_scale = max_abs(right);
            if left_scale == 0.0 || right_scale == 0.0 {
                borrowed_metric(left, right)
            } else {
                PreparedMetric {
                    left: Cow::Owned(scale_vector(left, left_scale)),
                    right: Cow::Owned(scale_vector(right, right_scale)),
                    result_scale: left_scale * right_scale,
                }
            }
        }
        Metric::Hamming | Metric::Jaccard => borrowed_metric(left, right),
    }
}

pub(crate) fn finish_prepared_metric(
    metric: Metric,
    partials: &[[f32; 4]],
    prepared: &PreparedMetric<'_>,
) -> Result<f32, String> {
    let value = f64::from(finish_metric(metric, partials)?) * prepared.result_scale;
    checked_f32(value)
}

pub(crate) fn prepare_normalization(
    vector: &[f32],
    method: u8,
) -> Result<PreparedNormalization, String> {
    match method {
        1 => prepare_l2_normalization(vector),
        2 => prepare_zscore_normalization(vector),
        3 => prepare_minmax_normalization(vector),
        _ => Err("unknown normalization".to_string()),
    }
}

fn prepare_l2_normalization(vector: &[f32]) -> Result<PreparedNormalization, String> {
    let scale = max_abs(vector);
    if scale == 0.0 {
        return Ok(zero_normalization(vector.len()));
    }

    let vector = scale_vector(vector, scale);
    let norm = vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(PreparedNormalization {
        vector,
        first: checked_f32(norm)?,
        second: 0.0,
    })
}

fn prepare_zscore_normalization(vector: &[f32]) -> Result<PreparedNormalization, String> {
    let divisor = vector.len() as f64;
    let mean = vector.iter().map(|value| f64::from(*value)).sum::<f64>() / divisor;
    let centered = vector
        .iter()
        .map(|value| f64::from(*value) - mean)
        .collect::<Vec<_>>();
    let variance = centered.iter().map(|value| value * value).sum::<f64>() / divisor;
    let stddev = variance.sqrt();

    if stddev == 0.0 {
        return Ok(zero_normalization(vector.len()));
    }

    let scale = centered.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let vector = centered
        .into_iter()
        .map(|value| (value / scale) as f32)
        .collect();

    Ok(PreparedNormalization {
        vector,
        first: 0.0,
        second: checked_f32(stddev / scale)?,
    })
}

fn prepare_minmax_normalization(vector: &[f32]) -> Result<PreparedNormalization, String> {
    let minimum = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if minimum == maximum {
        return Ok(zero_normalization(vector.len()));
    }

    let scale = f64::from(minimum).abs().max(f64::from(maximum).abs());
    let range = f64::from(maximum) - f64::from(minimum);

    Ok(PreparedNormalization {
        vector: scale_vector(vector, scale),
        first: checked_f32(f64::from(minimum) / scale)?,
        second: checked_f32(range / scale)?,
    })
}

fn zero_normalization(length: usize) -> PreparedNormalization {
    PreparedNormalization {
        vector: vec![0.0; length],
        first: 0.0,
        second: 0.0,
    }
}

fn borrowed_metric<'a>(left: &'a [f32], right: &'a [f32]) -> PreparedMetric<'a> {
    PreparedMetric {
        left: Cow::Borrowed(left),
        right: Cow::Borrowed(right),
        result_scale: 1.0,
    }
}

fn scaled_or_borrowed(values: &[f32]) -> Cow<'_, [f32]> {
    let scale = max_abs(values);
    if scale == 0.0 {
        Cow::Borrowed(values)
    } else {
        Cow::Owned(scale_vector(values, scale))
    }
}

fn max_abs(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).abs())
        .fold(0.0, f64::max)
}

fn scale_vector(values: &[f32], scale: f64) -> Vec<f32> {
    values
        .iter()
        .map(|value| (f64::from(*value) / scale) as f32)
        .collect()
}

fn checked_f32(value: f64) -> Result<f32, String> {
    if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
        Ok(value as f32)
    } else {
        Err(distances::METRIC_OVERFLOW.to_string())
    }
}

pub(crate) fn selected_matrix_rows(
    matrix: &[u8],
    dimensions: usize,
    row_indices: &[usize],
) -> Result<Vec<u8>, String> {
    if dimensions == 0 {
        return Err("invalid dimensions".to_string());
    }
    if row_indices.is_empty() {
        return Err("empty row selection".to_string());
    }

    let row_bytes = dimensions
        .checked_mul(4)
        .ok_or_else(|| "invalid dimensions".to_string())?;
    if matrix.is_empty() || !matrix.len().is_multiple_of(row_bytes) {
        return Err("matrix shape mismatch".to_string());
    }

    let row_count = matrix.len() / row_bytes;
    let selected_size = row_bytes
        .checked_mul(row_indices.len())
        .ok_or_else(|| "matrix shape mismatch".to_string())?;
    let mut selected_matrix = Vec::with_capacity(selected_size);

    for &row_index in row_indices {
        if row_index >= row_count {
            return Err("invalid row index".to_string());
        }
        let start = row_index * row_bytes;
        crate::dense::decode_f32_le(&matrix[start..start + row_bytes])?;
        selected_matrix.extend_from_slice(&matrix[start..start + row_bytes]);
    }

    Ok(selected_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_codes_and_reductions_cover_every_semantic() {
        let all_metrics = [
            Metric::L2,
            Metric::L2Squared,
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::NegativeInnerProduct,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Hamming,
            Metric::Jaccard,
        ];
        for (code, metric) in all_metrics.into_iter().enumerate() {
            assert_eq!(metric_code(metric), code as u32);
        }

        let partials = [[5.0, 25.0, 25.0, 0.0], [20.0, 0.0, 0.0, 0.0]];
        assert_eq!(finish_metric(Metric::L2, &partials), Ok(5.0));
        assert_eq!(finish_metric(Metric::L2Squared, &partials), Ok(25.0));
        assert_eq!(finish_metric(Metric::InnerProduct, &partials), Ok(25.0));
        assert_eq!(
            finish_metric(Metric::NegativeInnerProduct, &partials),
            Ok(-25.0)
        );
        assert_eq!(finish_metric(Metric::Manhattan, &partials), Ok(25.0));
        assert_eq!(finish_metric(Metric::Hamming, &partials), Ok(25.0));
        assert_eq!(finish_metric(Metric::Chebyshev, &partials), Ok(20.0));
        assert_eq!(finish_metric(Metric::Cosine, &partials), Ok(1.0));
        assert_eq!(finish_metric(Metric::Cosine, &[[0.0; 4]]), Ok(0.0));
        assert_eq!(
            finish_metric(Metric::Jaccard, &[[2.0, 4.0, 0.0, 0.0]]),
            Ok(0.5)
        );
        assert_eq!(finish_metric(Metric::Jaccard, &[[0.0; 4]]), Ok(0.0));
        assert!(finish_metric(Metric::L2Squared, &[[f32::INFINITY; 4]]).is_err());
    }

    #[test]
    fn metric_preparation_stabilizes_extreme_values() {
        let maximum = f32::MAX;

        let cosine_left = [2.0e19];
        let cosine_right = [1.0];
        let cosine = prepare_metric(&cosine_left, &cosine_right, Metric::Cosine);
        assert_eq!(cosine.left.as_ref(), &[1.0]);
        assert_eq!(cosine.right.as_ref(), &[1.0]);

        let dot_left = [maximum, maximum];
        let dot_right = [maximum, -maximum];
        let dot = prepare_metric(&dot_left, &dot_right, Metric::InnerProduct);
        assert_eq!(dot.left.as_ref(), &[1.0, 1.0]);
        assert_eq!(dot.right.as_ref(), &[1.0, -1.0]);
        assert_eq!(
            finish_prepared_metric(Metric::InnerProduct, &[[0.0; 4]], &dot),
            Ok(0.0)
        );

        let squared_left = [maximum];
        let squared_right = [0.0];
        let squared = prepare_metric(&squared_left, &squared_right, Metric::L2Squared);
        assert!(
            finish_prepared_metric(Metric::L2Squared, &[[1.0, 0.0, 0.0, 0.0]], &squared).is_err()
        );

        let zeros = [0.0, 0.0];
        let l2_zero = prepare_metric(&zeros, &zeros, Metric::L2);
        assert!(matches!(l2_zero.left, std::borrow::Cow::Borrowed(_)));
        let dot_zero = prepare_metric(&zeros, &dot_right, Metric::InnerProduct);
        assert!(matches!(dot_zero.left, std::borrow::Cow::Borrowed(_)));
        let cosine_zero = prepare_metric(&zeros, &dot_right, Metric::Cosine);
        assert!(matches!(cosine_zero.left, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn normalization_preparation_is_stable_for_small_and_offset_values() {
        let l2 = prepare_normalization(&[1.0e-23; 4], 1).unwrap();
        assert_eq!(l2.vector, vec![1.0; 4]);
        assert_eq!(l2.first, 2.0);

        let zscore = prepare_normalization(&[10_000.0, 10_000.1], 2).unwrap();
        assert_eq!(zscore.first, 0.0);
        assert!(zscore.second.is_finite() && zscore.second > 0.0);
        assert_eq!(zscore.vector, vec![-1.0, 1.0]);

        let minmax = prepare_normalization(&[-f32::MAX, f32::MAX], 3).unwrap();
        assert_eq!(minmax.vector, vec![-1.0, 1.0]);
        assert_eq!(minmax.first, -1.0);
        assert_eq!(minmax.second, 2.0);

        assert!(prepare_normalization(&[1.0], 99).is_err());
    }

    #[test]
    fn selected_rows_match_the_cpu_matrix_contract() {
        let matrix = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            selected_matrix_rows(&matrix, 2, &[0, 1]),
            Ok(matrix.clone())
        );
        assert_eq!(
            selected_matrix_rows(&matrix, 2, &[1, 1]),
            Ok([&matrix[8..], &matrix[8..]].concat())
        );
        assert!(selected_matrix_rows(&matrix, 0, &[0]).is_err());
        assert!(selected_matrix_rows(&matrix, usize::MAX, &[0]).is_err());
        assert!(selected_matrix_rows(&matrix, 2, &[]).is_err());
        assert!(selected_matrix_rows(&[], 2, &[0]).is_err());
        assert!(selected_matrix_rows(&matrix[..15], 2, &[0]).is_err());
        assert!(selected_matrix_rows(&matrix, 2, &[2]).is_err());

        let nan = f32::NAN.to_le_bytes();
        assert!(selected_matrix_rows(&nan, 1, &[0]).is_err());
    }
}
