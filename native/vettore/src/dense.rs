//! Dense f32 binary interoperability helpers.
//!
//! Elixir applications commonly persist embeddings as little-endian f32
//! binaries.  Keeping the decoding and pooling boundary here avoids creating
//! one BEAM term per coordinate before native vector work can begin.

use crate::distances::{self, Metric};

const INVALID_F32_BINARY: &str = "invalid f32 binary";
const INVALID_DIMENSIONS: &str = "invalid dimensions";
const INVALID_ROW_INDEX: &str = "invalid row index";
const EMPTY_SELECTION: &str = "empty row selection";
const MATRIX_SHAPE_MISMATCH: &str = "matrix shape mismatch";

/// Decodes one little-endian f32 vector and rejects malformed/non-finite data.
pub fn decode_f32_le(binary: &[u8]) -> Result<Vec<f32>, String> {
    if !binary.len().is_multiple_of(4) {
        return Err(INVALID_F32_BINARY.to_string());
    }

    let vector = binary
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();

    distances::validate_finite_vector(&vector)?;
    Ok(vector)
}

/// Computes any named metric directly over two little-endian f32 binaries.
pub fn metric_f32_le(left: &[u8], right: &[u8], metric: Metric) -> Result<f32, String> {
    let left = decode_f32_le(left)?;
    let right = decode_f32_le(right)?;
    distances::compute(metric, &left, &right)
}

/// Normalizes a little-endian f32 binary and returns native f32 coordinates.
pub fn normalize_f32_le(binary: &[u8], method: u8) -> Result<Vec<f32>, String> {
    let vector = decode_f32_le(binary)?;

    match method {
        0 => Ok(vector),
        1 => distances::normalize_l2(vector),
        2 => distances::normalize_zscore(vector),
        3 => distances::normalize_minmax(vector),
        _ => Err("unknown normalization".to_string()),
    }
}

/// Mean-pools selected rows from a row-major little-endian f32 matrix.
///
/// Repeated row indices are intentionally counted repeatedly because token-id
/// sequences may contain the same token more than once.
pub fn mean_pool_f32_le(
    matrix: &[u8],
    dimensions: usize,
    row_indices: &[usize],
) -> Result<Vec<f32>, String> {
    if dimensions == 0 {
        return Err(INVALID_DIMENSIONS.to_string());
    }
    if row_indices.is_empty() {
        return Err(EMPTY_SELECTION.to_string());
    }

    let row_bytes = dimensions
        .checked_mul(4)
        .ok_or_else(|| INVALID_DIMENSIONS.to_string())?;

    if matrix.is_empty() || !matrix.len().is_multiple_of(row_bytes) {
        return Err(MATRIX_SHAPE_MISMATCH.to_string());
    }

    let row_count = matrix.len() / row_bytes;
    let mut sums = vec![0.0f64; dimensions];

    for &row_index in row_indices {
        if row_index >= row_count {
            return Err(INVALID_ROW_INDEX.to_string());
        }

        let start = row_index * row_bytes;
        let row = decode_f32_le(&matrix[start..start + row_bytes])?;

        for (sum, value) in sums.iter_mut().zip(row) {
            *sum += f64::from(value);
        }
    }

    let divisor = row_indices.len() as f64;
    sums.into_iter()
        .map(|sum| {
            let value = sum / divisor;
            if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
                Ok(value as f32)
            } else {
                Err(distances::METRIC_OVERFLOW.to_string())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn decodes_little_endian_f32_and_rejects_invalid_data() {
        assert_eq!(
            decode_f32_le(&encode(&[1.0, -2.5, 3.25])),
            Ok(vec![1.0, -2.5, 3.25])
        );
        assert_eq!(decode_f32_le(&[]), Ok(vec![]));
        assert_eq!(decode_f32_le(&[0, 1, 2]), Err(INVALID_F32_BINARY.into()));
        assert!(decode_f32_le(&encode(&[f32::NAN])).is_err());
        assert!(decode_f32_le(&encode(&[f32::INFINITY])).is_err());
    }

    #[test]
    fn computes_metrics_and_normalizations_from_binary() {
        let left = encode(&[3.0, 4.0]);
        let right = encode(&[6.0, 8.0]);

        assert_eq!(metric_f32_le(&left, &right, Metric::Cosine), Ok(1.0));
        assert_eq!(metric_f32_le(&left, &right, Metric::InnerProduct), Ok(50.0));
        assert_eq!(normalize_f32_le(&left, 0), Ok(vec![3.0, 4.0]));
        assert_eq!(normalize_f32_le(&left, 1), Ok(vec![0.6, 0.8]));
        assert!(normalize_f32_le(&left, 2)
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(normalize_f32_le(&left, 3), Ok(vec![0.0, 1.0]));
        assert!(normalize_f32_le(&left, 99).is_err());
        assert!(metric_f32_le(&left, &encode(&[1.0]), Metric::L2).is_err());
    }

    #[test]
    fn mean_pool_supports_order_duplicates_and_shape_validation() {
        let matrix = encode(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(
            mean_pool_f32_le(&matrix, 3, &[0, 2]),
            Ok(vec![4.0, 5.0, 6.0])
        );
        assert_eq!(
            mean_pool_f32_le(&matrix, 3, &[1, 1, 2]),
            Ok(vec![5.0, 6.0, 7.0])
        );
        assert!(mean_pool_f32_le(&matrix, 0, &[0]).is_err());
        assert!(mean_pool_f32_le(&matrix, 3, &[]).is_err());
        assert!(mean_pool_f32_le(&matrix[..matrix.len() - 1], 3, &[0]).is_err());
        assert!(mean_pool_f32_le(&matrix, 3, &[3]).is_err());
    }
}
