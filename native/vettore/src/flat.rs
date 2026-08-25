//! Native exact flat index resource.
//!
//! ETS remains the canonical record store. This resource mirrors ids and one
//! contiguous row-major matrix for cache-friendly SIMD scans. An optional,
//! generation-aware GPU snapshot keeps the stable-id matrix resident and runs
//! batched scoring plus two-stage top-k entirely on the device.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};

use crate::distances::Metric;

pub struct FlatIndex {
    metric: Metric,
    ids: Vec<String>,
    vectors: Vec<f32>,
    positions: HashMap<String, usize>,
    dimension: Option<usize>,
    generation: u64,
}

#[derive(Debug)]
struct FlatHit {
    id: String,
    raw: f32,
    rank: f32,
}

impl Eq for FlatHit {}

impl PartialEq for FlatHit {
    fn eq(&self, other: &Self) -> bool {
        self.rank.total_cmp(&other.rank) == Ordering::Equal && self.id == other.id
    }
}

impl Ord for FlatHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .total_cmp(&other.rank)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for FlatHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FlatIndex {
    /// Creates an empty exact flat index for one metric.
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
            positions: HashMap::new(),
            dimension: None,
            generation: 0,
        }
    }

    /// Inserts or replaces one vector by external id.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<(), String> {
        self.validate_vector(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }
        self.insert_validated(id, vector);
        self.bump_generation();
        Ok(())
    }

    /// Inserts or replaces a batch of vectors.
    pub fn insert_many(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<(), String> {
        let expected = self
            .dimension
            .or_else(|| vectors.first().map(|(_, vector)| vector.len()));

        for (_, vector) in &vectors {
            validate_vector(vector, expected)?;
        }

        let changed = !vectors.is_empty();
        if self.dimension.is_none() {
            self.dimension = expected;
        }
        for (id, vector) in vectors {
            self.insert_validated(id, vector);
        }
        if changed {
            self.bump_generation();
        }
        Ok(())
    }

    /// Deletes one vector by external id.
    pub fn delete(&mut self, id: &str) {
        let Some(position) = self.positions.remove(id) else {
            return;
        };
        let dimension = self
            .dimension
            .expect("non-empty flat index has a dimension");
        let last = self.ids.len() - 1;

        self.ids.swap_remove(position);
        if position != last {
            let source = last * dimension..(last + 1) * dimension;
            self.vectors.copy_within(source, position * dimension);
            self.positions.insert(self.ids[position].clone(), position);
        }
        self.vectors.truncate(last * dimension);

        if self.ids.is_empty() {
            self.dimension = None;
        }
        self.bump_generation();
    }

    /// Releases all mirrored vectors while keeping the resource reusable.
    pub fn clear(&mut self) {
        if self.ids.is_empty() {
            return;
        }
        self.ids.clear();
        self.vectors.clear();
        self.positions.clear();
        self.dimension = None;
        self.bump_generation();
    }

    /// Searches every stored vector and returns ids with raw metric values.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        validate_vector(query, self.dimension)?;

        let dimension = self.dimension.unwrap_or(query.len());
        let mut hits = BinaryHeap::with_capacity(usize::min(limit, self.ids.len()));
        for (row, vector) in self.vectors.chunks_exact(dimension).enumerate() {
            let raw = match crate::distances::compute(self.metric, query, vector) {
                Ok(raw) => raw,
                Err(reason) if crate::distances::is_metric_overflow(&reason) => continue,
                Err(reason) => return Err(reason),
            };
            let hit = FlatHit {
                id: self.ids[row].clone(),
                raw,
                rank: crate::distances::rank_value(self.metric, raw),
            };

            if hits.len() < limit {
                hits.push(hit);
            } else if hits.peek().is_some_and(|worst| hit < *worst) {
                hits.pop();
                hits.push(hit);
            }
        }

        let mut hits = hits.into_vec();
        hits.sort();

        Ok(hits.into_iter().map(|hit| (hit.id, hit.raw)).collect())
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), String> {
        validate_vector(vector, self.dimension)
    }

    fn insert_validated(&mut self, id: String, vector: Vec<f32>) {
        let dimension = self
            .dimension
            .expect("validated flat vector has a dimension");
        if let Some(&position) = self.positions.get(&id) {
            let start = position * dimension;
            self.vectors[start..start + dimension].copy_from_slice(&vector);
            return;
        }

        let position = self.ids.len();
        self.positions.insert(id.clone(), position);
        self.ids.push(id);
        self.vectors.extend(vector);
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn workload(&self) -> (usize, usize) {
        (self.ids.len(), self.dimension.unwrap_or(0))
    }

    fn gpu_snapshot(&self) -> FlatSnapshot {
        let dimension = self.dimension.unwrap_or(0);
        FlatSnapshot {
            generation: self.generation,
            metric: self.metric,
            dimension,
            ids: self.ids.clone(),
            vectors: self.vectors.clone(),
        }
    }
}

struct FlatSnapshot {
    generation: u64,
    metric: Metric,
    dimension: usize,
    ids: Vec<String>,
    vectors: Vec<f32>,
}

impl FlatSnapshot {
    fn sort_by_id(mut self) -> Self {
        let rows = self.ids.len();
        let mut desired_old_at_position = (0..rows).collect::<Vec<_>>();
        desired_old_at_position
            .sort_unstable_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));

        let mut old_at_position = (0..rows).collect::<Vec<_>>();
        let mut position_of_old = (0..rows).collect::<Vec<_>>();
        for new_position in 0..rows {
            let desired_old = desired_old_at_position[new_position];
            let current_position = position_of_old[desired_old];
            if current_position == new_position {
                continue;
            }

            let displaced_old = old_at_position[new_position];
            self.ids.swap(new_position, current_position);
            swap_matrix_rows(
                &mut self.vectors,
                self.dimension,
                new_position,
                current_position,
            );
            old_at_position.swap(new_position, current_position);
            position_of_old[desired_old] = new_position;
            position_of_old[displaced_old] = current_position;
        }

        self
    }
}

fn swap_matrix_rows(matrix: &mut [f32], dimension: usize, left: usize, right: usize) {
    if left == right || dimension == 0 {
        return;
    }
    let (left, right) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    let right_start = right * dimension;
    let (before_right, from_right) = matrix.split_at_mut(right_start);
    before_right[left * dimension..(left + 1) * dimension]
        .swap_with_slice(&mut from_right[..dimension]);
}

struct FlatGpuCache {
    generation: u64,
    ids: Vec<String>,
    matrix: crate::gpu::ResidentMatrix,
}

enum FlatGpuCacheState {
    Empty,
    Ready(Arc<FlatGpuCache>),
    Failed { generation: u64, error: String },
}

pub struct FlatResource {
    index: RwLock<FlatIndex>,
    gpu_cache: Mutex<FlatGpuCacheState>,
    gpu_build: Mutex<()>,
    gpu_builds: AtomicU64,
}

impl FlatResource {
    pub fn new(metric: Metric) -> Self {
        Self {
            index: RwLock::new(FlatIndex::new(metric)),
            gpu_cache: Mutex::new(FlatGpuCacheState::Empty),
            gpu_build: Mutex::new(()),
            gpu_builds: AtomicU64::new(0),
        }
    }

    pub fn insert(&self, id: String, vector: Vec<f32>) -> Result<(), String> {
        let mut index = self.write_index()?;
        index.insert(id, vector)?;
        self.clear_gpu_cache();
        Ok(())
    }

    pub fn insert_many(&self, vectors: Vec<(String, Vec<f32>)>) -> Result<(), String> {
        let changed = !vectors.is_empty();
        let mut index = self.write_index()?;
        index.insert_many(vectors)?;
        if changed {
            self.clear_gpu_cache();
        }
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let mut index = self.write_index()?;
        let generation = index.generation;
        index.delete(id);
        if index.generation != generation {
            self.clear_gpu_cache();
        }
        Ok(())
    }

    pub fn clear(&self) -> Result<(), String> {
        let mut index = self.write_index()?;
        let generation = index.generation;
        index.clear();
        if index.generation != generation {
            self.clear_gpu_cache();
        }
        Ok(())
    }

    pub fn workload(&self) -> Result<(usize, usize), String> {
        Ok(self.read_index()?.workload())
    }

    pub fn gpu_cache_info(&self) -> Result<(u64, bool), String> {
        let state = self
            .gpu_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cached = matches!(
            &*state,
            FlatGpuCacheState::Ready(cache) if cache.matrix.is_current()
        );
        Ok((self.gpu_builds.load(AtomicOrdering::Relaxed), cached))
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        self.read_index()?.search(query, limit)
    }

    pub fn search_gpu(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let Some(generation) = self.validate_gpu_search(query, limit)? else {
            return Ok(Vec::new());
        };

        let cache = match self.cached_gpu_generation(generation) {
            Some(result) => result?,
            None => match self.build_gpu_cache(query, limit)? {
                Some(cache) => cache,
                None => return Ok(Vec::new()),
            },
        };
        let hits = cache.matrix.search(query, limit)?;
        Ok(hits
            .into_iter()
            .map(|(row, raw)| (cache.ids[row].clone(), raw))
            .collect())
    }

    fn validate_gpu_search(&self, query: &[f32], limit: usize) -> Result<Option<u64>, String> {
        let index = self.read_index()?;
        validate_vector(query, index.dimension)?;
        if index.ids.is_empty() {
            return Ok(None);
        }
        if usize::min(limit, index.ids.len()) > crate::gpu::MAX_RESIDENT_TOP_K as usize {
            return Err(format!(
                "gpu flat top-k supports at most {} results",
                crate::gpu::MAX_RESIDENT_TOP_K
            ));
        }
        let generation = index.generation;
        let metric = index.metric;
        drop(index);
        crate::gpu::validate_resident_query(query, metric)?;
        Ok(Some(generation))
    }

    fn cached_gpu_generation(&self, generation: u64) -> Option<Result<Arc<FlatGpuCache>, String>> {
        let mut state = self
            .gpu_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*state {
            FlatGpuCacheState::Ready(cache)
                if cache.generation == generation && cache.matrix.is_current() =>
            {
                Some(Ok(Arc::clone(cache)))
            }
            FlatGpuCacheState::Ready(cache) if cache.generation == generation => {
                *state = FlatGpuCacheState::Empty;
                None
            }
            FlatGpuCacheState::Failed {
                generation: failed_generation,
                error,
            } if *failed_generation == generation => Some(Err(error.clone())),
            FlatGpuCacheState::Empty
            | FlatGpuCacheState::Ready(_)
            | FlatGpuCacheState::Failed { .. } => None,
        }
    }

    fn build_gpu_cache(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Option<Arc<FlatGpuCache>>, String> {
        let _build = self
            .gpu_build
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let Some(generation) = self.validate_gpu_search(query, limit)? else {
            return Ok(None);
        };
        if let Some(cached) = self.cached_gpu_generation(generation) {
            return cached.map(Some);
        }

        let snapshot = {
            let index = self.read_index()?;
            validate_vector(query, index.dimension)?;
            if index.ids.is_empty() {
                return Ok(None);
            }
            if usize::min(limit, index.ids.len()) > crate::gpu::MAX_RESIDENT_TOP_K as usize {
                return Err(format!(
                    "gpu flat top-k supports at most {} results",
                    crate::gpu::MAX_RESIDENT_TOP_K
                ));
            }
            index.gpu_snapshot()
        };
        crate::gpu::validate_resident_query(query, snapshot.metric)?;
        let snapshot = snapshot.sort_by_id();
        let result = crate::gpu::resident_matrix(
            snapshot.vectors,
            snapshot.ids.len(),
            snapshot.dimension,
            snapshot.metric,
        );

        match result {
            Ok(matrix) => {
                let built = Arc::new(FlatGpuCache {
                    generation: snapshot.generation,
                    ids: snapshot.ids,
                    matrix,
                });
                self.gpu_builds.fetch_add(1, AtomicOrdering::Relaxed);
                self.publish_gpu_state(
                    snapshot.generation,
                    FlatGpuCacheState::Ready(Arc::clone(&built)),
                )?;
                Ok(Some(built))
            }
            Err(error) => {
                if cacheable_gpu_build_failure(&error) {
                    self.publish_gpu_state(
                        snapshot.generation,
                        FlatGpuCacheState::Failed {
                            generation: snapshot.generation,
                            error: error.clone(),
                        },
                    )?;
                }
                Err(error)
            }
        }
    }

    fn publish_gpu_state(&self, generation: u64, state: FlatGpuCacheState) -> Result<(), String> {
        let index = self.read_index()?;
        if index.generation == generation {
            *self
                .gpu_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = state;
        }
        Ok(())
    }

    fn clear_gpu_cache(&self) {
        *self
            .gpu_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = FlatGpuCacheState::Empty;
    }

    fn read_index(&self) -> Result<std::sync::RwLockReadGuard<'_, FlatIndex>, String> {
        self.index
            .read()
            .map_err(|_| "flat lock poisoned".to_string())
    }

    fn write_index(&self) -> Result<std::sync::RwLockWriteGuard<'_, FlatIndex>, String> {
        self.index
            .write()
            .map_err(|_| "flat lock poisoned".to_string())
    }
}

#[rustler::resource_impl]
impl rustler::Resource for FlatResource {}

fn validate_vector(vector: &[f32], dimension: Option<usize>) -> Result<(), String> {
    if vector.is_empty() {
        return Err("vector must not be empty".to_string());
    }
    if dimension.is_some_and(|expected| vector.len() != expected) {
        return Err("dimension mismatch".to_string());
    }
    crate::distances::validate_finite_vector(vector)
}

fn cacheable_gpu_build_failure(error: &str) -> bool {
    error == "gpu numeric range unsupported" || error == "gpu workload too large"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_metrics() -> [Metric; 9] {
        [
            Metric::L2,
            Metric::L2Squared,
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::NegativeInnerProduct,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Hamming,
            Metric::Jaccard,
        ]
    }

    #[test]
    fn inserts_replaces_deletes_and_returns_stable_top_k() {
        let mut index = FlatIndex::new(Metric::L2);
        index.insert("b".into(), vec![2.0]).unwrap();
        index.insert("a".into(), vec![0.0]).unwrap();
        index.insert("c".into(), vec![2.0]).unwrap();

        assert_eq!(
            index.search(&[1.0], 2).unwrap(),
            vec![("a".into(), 1.0), ("b".into(), 1.0)]
        );

        index.insert("a".into(), vec![10.0]).unwrap();
        assert_eq!(index.search(&[2.0], 1).unwrap()[0].0, "b");
        index.delete("b");
        assert_eq!(index.search(&[2.0], 1).unwrap()[0].0, "c");
    }

    #[test]
    fn batch_validation_is_atomic() {
        let mut index = FlatIndex::new(Metric::InnerProduct);
        index.insert("existing".into(), vec![1.0, 0.0]).unwrap();

        assert!(index
            .insert_many(vec![
                ("valid".into(), vec![0.0, 1.0]),
                ("invalid".into(), vec![1.0]),
            ])
            .is_err());
        assert_eq!(index.ids.len(), 1);
        assert!(!index.positions.contains_key("valid"));
        assert!(index.insert("nan".into(), vec![f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn rejects_invalid_queries_and_handles_empty_limits() {
        let mut index = FlatIndex::new(Metric::Cosine);
        assert!(index.insert("empty".into(), vec![]).is_err());
        index.insert("a".into(), vec![1.0, 0.0]).unwrap();
        assert!(index.search(&[1.0], 1).is_err());
        assert!(index.search(&[f32::INFINITY, 0.0], 1).is_err());
        assert_eq!(index.search(&[1.0, 0.0], 0).unwrap(), Vec::new());
    }

    #[test]
    fn exact_heap_matches_a_full_sort_for_all_metrics() {
        let vectors: Vec<_> = (0..51)
            .map(|index| {
                (
                    format!("v-{index:02}"),
                    vec![
                        (index as f32 - 25.0) / 9.0,
                        ((index * 13 % 31) as f32 - 15.0) / 7.0,
                        if index % 2 == 0 { 0.0 } else { 1.0 },
                    ],
                )
            })
            .collect();
        let query = [0.5, -1.25, 1.0];

        for metric in all_metrics() {
            let mut index = FlatIndex::new(metric);
            index.insert_many(vectors.clone()).unwrap();

            let mut expected: Vec<_> = vectors
                .iter()
                .map(|(id, vector)| {
                    (
                        id.clone(),
                        crate::distances::compute(metric, &query, vector).unwrap(),
                    )
                })
                .collect();
            expected.sort_by(|left, right| {
                crate::distances::rank_value(metric, left.1)
                    .total_cmp(&crate::distances::rank_value(metric, right.1))
                    .then_with(|| left.0.cmp(&right.0))
            });

            for limit in [1usize, 7, 51, 100] {
                let mut limited = expected.clone();
                limited.truncate(limit);
                assert_eq!(index.search(&query, limit).unwrap(), limited);
            }
        }
    }

    #[test]
    fn empty_batches_unknown_deletes_and_dimension_resets_are_total() {
        let mut index = FlatIndex::new(Metric::L2);
        assert_eq!(index.insert_many(vec![]), Ok(()));
        assert_eq!(index.search(&[1.0], 10), Ok(vec![]));
        index.delete("missing");

        index.insert("one".into(), vec![1.0]).unwrap();
        index.delete("missing");
        assert_eq!(index.dimension, Some(1));
        index.delete("one");
        assert_eq!(index.dimension, None);

        index.insert("two".into(), vec![1.0, 2.0]).unwrap();
        assert_eq!(index.dimension, Some(2));
        assert_eq!(index.search(&[1.0, 2.0], usize::MAX).unwrap().len(), 1);
        index.clear();
        assert!(index.vectors.is_empty());
        assert!(index.ids.is_empty());
        assert_eq!(index.dimension, None);
    }

    #[test]
    fn duplicate_batch_ids_replace_deterministically_and_large_l2_stays_finite() {
        let mut index = FlatIndex::new(Metric::L2);
        index
            .insert_many(vec![
                ("same".into(), vec![0.0]),
                ("same".into(), vec![1.0e20]),
            ])
            .unwrap();
        assert_eq!(index.ids.len(), 1);
        assert_eq!(index.search(&[0.0], 1).unwrap()[0].0, "same");
        assert!(index.search(&[0.0], 1).unwrap()[0].1.is_finite());
    }

    #[test]
    fn search_skips_only_rows_whose_score_overflows() {
        let mut index = FlatIndex::new(Metric::L2);
        index.insert("safe".into(), vec![0.0]).unwrap();
        index.insert("overflow".into(), vec![-f32::MAX]).unwrap();

        assert_eq!(
            index.search(&[f32::MAX], 2).unwrap(),
            vec![("safe".into(), f32::MAX)]
        );
    }

    #[test]
    fn heap_hit_equality_and_partial_order_include_the_external_id() {
        let first = FlatHit {
            id: "a".into(),
            raw: 1.0,
            rank: 1.0,
        };
        let equal = FlatHit {
            id: "a".into(),
            raw: 99.0,
            rank: 1.0,
        };
        let other_id = FlatHit {
            id: "b".into(),
            raw: 1.0,
            rank: 1.0,
        };
        assert_eq!(first, equal);
        assert_ne!(first, other_id);
        assert_eq!(first.partial_cmp(&other_id), Some(Ordering::Less));
    }

    #[test]
    fn dense_rows_positions_generations_and_gpu_snapshots_stay_consistent() {
        let mut index = FlatIndex::new(Metric::L2);
        assert_eq!(index.workload(), (0, 0));
        assert_eq!(index.generation, 0);

        index.insert("c".into(), vec![3.0, 30.0]).unwrap();
        index.insert("a".into(), vec![1.0, 10.0]).unwrap();
        index.insert("b".into(), vec![2.0, 20.0]).unwrap();
        index.insert("a".into(), vec![1.5, 15.0]).unwrap();
        assert_eq!(index.workload(), (3, 2));
        assert_eq!(index.vectors.len(), 6);
        assert_eq!(index.generation, 4);

        index.delete("c");
        assert_eq!(index.workload(), (2, 2));
        assert_eq!(index.vectors.len(), 4);
        for (position, id) in index.ids.iter().enumerate() {
            assert_eq!(index.positions[id], position);
        }

        let snapshot = index.gpu_snapshot().sort_by_id();
        assert_eq!(snapshot.ids, ["a", "b"]);
        assert_eq!(snapshot.vectors, [1.5, 15.0, 2.0, 20.0]);
        assert_eq!(snapshot.generation, index.generation);

        let hits = index.search(&[1.5, 15.0], 2).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(hits[0].1, 0.0);
        assert!((hits[1].1 - 5.024_937_6).abs() < 1.0e-6);
    }

    #[test]
    fn deterministic_gpu_build_failures_are_cached_until_the_generation_changes() {
        let resource = FlatResource::new(Metric::L2);
        resource
            .insert("unsafe".into(), vec![1.0e38, 1.0e-7])
            .unwrap();

        for _attempt in 0..2 {
            assert_eq!(
                resource.search_gpu(&[1.0, 1.0], 1),
                Err("gpu numeric range unsupported".into())
            );
        }
        assert_eq!(resource.gpu_cache_info(), Ok((0, false)));
        assert!(matches!(
            &*resource
                .gpu_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            FlatGpuCacheState::Failed { generation: 1, .. }
        ));

        resource.insert("unsafe".into(), vec![1.0, 2.0]).unwrap();
        assert!(matches!(
            &*resource
                .gpu_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            FlatGpuCacheState::Empty
        ));
    }

    #[test]
    fn flat_resource_wrappers_and_gpu_cache_follow_the_index_generation() {
        let resource = FlatResource::new(Metric::L2);
        assert_eq!(resource.workload(), Ok((0, 0)));
        assert_eq!(resource.insert_many(vec![]), Ok(()));
        assert_eq!(resource.search_gpu(&[1.0], 0), Ok(vec![]));
        assert_eq!(resource.search_gpu(&[1.0], 1), Ok(vec![]));

        resource
            .insert_many(vec![
                ("d".into(), vec![3.0]),
                ("b".into(), vec![1.0]),
                ("e".into(), vec![4.0]),
                ("a".into(), vec![0.0]),
                ("c".into(), vec![2.0]),
            ])
            .unwrap();
        assert_eq!(resource.workload(), Ok((5, 1)));
        assert_eq!(
            resource.search(&[0.0], 2),
            Ok(vec![("a".into(), 0.0), ("b".into(), 1.0)])
        );
        assert!(resource.search_gpu(&[0.0, 1.0], 2).is_err());

        if !crate::gpu::detected() {
            assert_ne!(std::env::var("VETTORE_REQUIRE_GPU").as_deref(), Ok("1"));
            return;
        }

        assert_eq!(
            resource.search_gpu(&[0.0], 2),
            Ok(vec![("a".into(), 0.0), ("b".into(), 1.0)])
        );
        assert_eq!(resource.gpu_cache_info(), Ok((1, true)));
        assert_eq!(resource.search_gpu(&[0.0], 1), Ok(vec![("a".into(), 0.0)]));
        assert_eq!(resource.gpu_cache_info(), Ok((1, true)));

        resource.delete("missing").unwrap();
        assert_eq!(resource.gpu_cache_info(), Ok((1, true)));
        resource.delete("a").unwrap();
        assert_eq!(resource.gpu_cache_info(), Ok((1, false)));
        assert_eq!(resource.search_gpu(&[0.0], 1), Ok(vec![("b".into(), 1.0)]));
        assert_eq!(resource.gpu_cache_info(), Ok((2, true)));

        resource.clear().unwrap();
        assert_eq!(resource.workload(), Ok((0, 0)));
        assert_eq!(resource.gpu_cache_info(), Ok((2, false)));
        resource.clear().unwrap();

        let large_window = FlatResource::new(Metric::L2);
        large_window
            .insert_many(
                (0..65)
                    .map(|row| (format!("row-{row:02}"), vec![row as f32]))
                    .collect(),
            )
            .unwrap();
        assert_eq!(
            large_window.search_gpu(&[0.0], 65),
            Err("gpu flat top-k supports at most 64 results".into())
        );
    }
}
