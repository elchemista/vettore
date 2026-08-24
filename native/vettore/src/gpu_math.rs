//! Pure validation and reduction helpers shared by the wgpu runtime.

use crate::distances::{self, Metric};

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

pub(crate) fn normalization_parameters(
    method: u8,
    length: usize,
    partials: &[[f32; 4]],
) -> Result<(f32, f32), String> {
    let sum = partials
        .iter()
        .map(|partial| f64::from(partial[0]))
        .sum::<f64>();
    let sum_squares = partials
        .iter()
        .map(|partial| f64::from(partial[1]))
        .sum::<f64>();
    let minimum = partials
        .iter()
        .map(|partial| partial[2])
        .fold(f32::INFINITY, f32::min);
    let maximum = partials
        .iter()
        .map(|partial| partial[3])
        .fold(f32::NEG_INFINITY, f32::max);

    match method {
        1 => Ok((sum_squares.sqrt() as f32, 0.0)),
        2 => {
            let divisor = length as f64;
            let mean = sum / divisor;
            let variance = (sum_squares / divisor - mean * mean).max(0.0);
            Ok((mean as f32, variance.sqrt() as f32))
        }
        3 => Ok((minimum, maximum - minimum)),
        _ => Err("unknown normalization".to_string()),
    }
}

pub(crate) fn cpu_normalize(vector: Vec<f32>, method: u8) -> Result<Vec<f32>, String> {
    match method {
        0 => Ok(vector),
        1 => distances::normalize_l2(vector),
        2 => distances::normalize_zscore(vector),
        3 => distances::normalize_minmax(vector),
        _ => Err("unknown normalization".to_string()),
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
    fn normalization_parameters_and_cpu_fallback_cover_every_method() {
        let partials = [[3.0, 5.0, 1.0, 2.0], [7.0, 25.0, 3.0, 4.0]];
        let (norm, _) = normalization_parameters(1, 4, &partials).unwrap();
        assert!((norm - 30.0f32.sqrt()).abs() < 1.0e-6);

        let (mean, stddev) = normalization_parameters(2, 4, &partials).unwrap();
        assert_eq!(mean, 2.5);
        assert!((stddev - 1.118_034).abs() < 1.0e-6);
        assert_eq!(normalization_parameters(3, 4, &partials), Ok((1.0, 3.0)));
        assert!(normalization_parameters(99, 4, &partials).is_err());

        assert_eq!(cpu_normalize(vec![3.0, 4.0], 0), Ok(vec![3.0, 4.0]));
        assert_eq!(cpu_normalize(vec![3.0, 4.0], 1), Ok(vec![0.6, 0.8]));
        assert!(cpu_normalize(vec![1.0, 2.0], 2).is_ok());
        assert_eq!(cpu_normalize(vec![1.0, 2.0], 3), Ok(vec![0.0, 1.0]));
        assert!(cpu_normalize(vec![1.0], 99).is_err());
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
