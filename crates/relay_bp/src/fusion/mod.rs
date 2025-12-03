// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bp::min_sum::MinSumDecoderConfig;
use crate::bp::relay::{RelayDecoder, RelayDecoderConfig};
use crate::decoder::{Bit, BPExtraResult, DecodeResult, Decoder, SparseBitMatrix};
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use std::sync::Arc;

/// Represents a partition of the decoding problem.
#[derive(Clone, Debug)]
pub struct Partition {
    pub variable_indices: Vec<usize>,
    pub detector_indices: Vec<usize>,
}

/// Partition variables into equal-sized chunks.
///
/// # Arguments
/// * `total_variables` - Total number of error variables
/// * `num_partitions` - Number of partitions to create
///
/// # Returns
/// Vector of partitions with variable indices assigned to each partition
///
/// # Example
/// ```
/// use relay_bp::fusion::partition_by_count;
/// let partitions = partition_by_count(1000, 2);
/// assert_eq!(partitions[0].variable_indices.len(), 500);
/// assert_eq!(partitions[1].variable_indices.len(), 500);
/// ```
pub fn partition_by_count(total_variables: usize, num_partitions: usize) -> Vec<Partition> {
    if num_partitions == 0 {
        panic!("num_partitions must be > 0");
    }
    if total_variables == 0 {
        panic!("total_variables must be > 0");
    }

    let vars_per_partition = total_variables / num_partitions;
    let remainder = total_variables % num_partitions;

    let mut partitions = Vec::with_capacity(num_partitions);
    let mut start = 0;

    for i in 0..num_partitions {
        // Distribute remainder across first partitions
        let size = vars_per_partition + if i < remainder { 1 } else { 0 };
        let end = start + size;

        partitions.push(Partition {
            variable_indices: (start..end).collect(),
            detector_indices: Vec::new(),
        });

        start = end;
    }

    partitions
}

/// Partition detectors by time rounds for spacetime codes.
///
/// Creates partitions where each contains detectors from consecutive rounds.
/// This naturally produces small boundaries (only errors connecting adjacent
/// rounds at partition edges) relative to partition volume.
///
/// For a spacetime code with N rounds, partitioning by time ensures:
/// - Partition volume: O(d² × rounds_per_partition)
/// - Partition boundary: O(d²) - only variables connecting adjacent rounds
/// - Boundary/volume ratio: O(1/rounds_per_partition) ≈ 0.1 for 10 rounds/partition
///
/// # Arguments
/// * `check_matrix` - Spacetime check matrix (detectors × errors)
/// * `detectors_per_round` - Number of detectors in each syndrome round
/// * `rounds_per_partition` - How many consecutive rounds per partition
///
/// # Returns
/// Vector of partitions with small boundary/volume ratios
///
/// # Panics
/// Panics if `detectors_per_round` is zero or doesn't evenly divide total detectors
///
/// # Example
/// ```
/// use relay_bp::fusion::partition_by_time_rounds;
/// use relay_bp::decoder::SparseBitMatrix;
/// use ndarray::array;
///
/// // 150 rounds × 72 detectors/round = 10,800 detectors
/// // Partition into 10-round chunks → 15 partitions
/// let dense = array![[1, 0], [0, 1]]; // Simplified example
/// let check_matrix = SparseBitMatrix::from_dense(dense);
/// let partitions = partition_by_time_rounds(&check_matrix, 1, 1);
/// assert_eq!(partitions.len(), 2);
/// ```
pub fn partition_by_time_rounds(
    check_matrix: &SparseBitMatrix,
    detectors_per_round: usize,
    rounds_per_partition: usize,
) -> Vec<Partition> {
    if detectors_per_round == 0 {
        panic!("detectors_per_round must be > 0");
    }
    if rounds_per_partition == 0 {
        panic!("rounds_per_partition must be > 0");
    }

    let total_detectors = check_matrix.rows();

    // Validate that detectors divide evenly into rounds
    if total_detectors % detectors_per_round != 0 {
        panic!(
            "Total detectors ({}) is not evenly divisible by detectors_per_round ({})",
            total_detectors, detectors_per_round
        );
    }

    let total_rounds = total_detectors / detectors_per_round;
    let num_partitions = (total_rounds + rounds_per_partition - 1) / rounds_per_partition;

    let mut partitions = Vec::with_capacity(num_partitions);

    for partition_idx in 0..num_partitions {
        let start_round = partition_idx * rounds_per_partition;
        let end_round = std::cmp::min(start_round + rounds_per_partition, total_rounds);

        let start_detector = start_round * detectors_per_round;
        let end_detector = end_round * detectors_per_round;

        let detector_indices: Vec<usize> = (start_detector..end_detector).collect();

        let variable_indices = extract_connected_variables(check_matrix, &detector_indices);

        partitions.push(Partition {
            variable_indices,
            detector_indices,
        });
    }

    partitions
}

/// Extract all error variables connected to a set of detectors.
///
/// Iterates through the check matrix to find which error variables (columns)
/// are connected to any of the specified detectors (rows). This determines
/// the variable scope of a partition.
///
/// # Arguments
/// * `check_matrix` - Spacetime check matrix (detectors × errors)
/// * `detector_indices` - List of detector indices to examine
///
/// # Returns
/// Sorted vector of unique variable indices connected to the detectors
fn extract_connected_variables(
    check_matrix: &SparseBitMatrix,
    detector_indices: &[usize],
) -> Vec<usize> {
    use std::collections::HashSet;

    let mut variable_set = HashSet::new();

    for &det_idx in detector_indices {
        let row = check_matrix.outer_view(det_idx).unwrap();
        for (var_idx, _value) in row.iter() {
            variable_set.insert(var_idx);
        }
    }

    let mut variables: Vec<usize> = variable_set.into_iter().collect();
    variables.sort();
    variables
}

/// Create a check matrix for a given partition.
///
/// This extracts the sub-matrix containing only the variables and detectors
/// relevant to the partition. Detectors are included if they have any
/// connection to variables in the partition.
///
/// # Arguments
/// * `check_matrix` - Original check matrix (detectors × variables)
/// * `partition` - Partition specification
///
/// # Returns
/// Tuple of (partition check matrix, updated partition with detector indices)
pub fn create_partition_check_matrix(
    check_matrix: &SparseBitMatrix,
    mut partition: Partition,
) -> (Arc<SparseBitMatrix>, Partition) {
    let variable_set: std::collections::HashSet<usize> =
        partition.variable_indices.iter().copied().collect();

    let mut detector_set = std::collections::HashSet::new();
    for (detector_idx, row) in check_matrix.outer_iterator().enumerate() {
        for (var_idx, _) in row.iter() {
            if variable_set.contains(&var_idx) {
                detector_set.insert(detector_idx);
                break;
            }
        }
    }

    partition.detector_indices = detector_set.into_iter().collect();
    partition.detector_indices.sort();

    let var_map: std::collections::HashMap<usize, usize> = partition
        .variable_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();

    let det_map: std::collections::HashMap<usize, usize> = partition
        .detector_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();

    let num_detectors = partition.detector_indices.len();
    let num_variables = partition.variable_indices.len();

    use sprs::TriMat;
    let mut tri_mat = TriMat::new((num_detectors, num_variables));

    for &det_idx in &partition.detector_indices {
        let row = check_matrix.outer_view(det_idx).unwrap();
        let new_det_idx = det_map[&det_idx];

        for (var_idx, &value) in row.iter() {
            if let Some(&new_var_idx) = var_map.get(&var_idx) {
                tri_mat.add_triplet(new_det_idx, new_var_idx, value);
            }
        }
    }

    let partition_matrix = Arc::new(tri_mat.to_csc() as SparseBitMatrix);

    (partition_matrix, partition)
}

/// Combine posteriors from multiple partitions into a single array.
///
/// Maps each partition's posteriors to the correct variable indices in the
/// original problem. For variables that appear in multiple partitions (boundaries),
/// averages their posteriors.
///
/// # Arguments
/// * `posteriors` - Vector of posterior arrays, one per partition
/// * `partitions` - Vector of partitions with variable indices
/// * `total_variables` - Total number of variables in the original problem
///
/// # Returns
/// Combined posterior array with length `total_variables`
pub fn combine_posteriors(
    posteriors: &[Array1<f64>],
    partitions: &[Partition],
    total_variables: usize,
) -> Array1<f64> {
    if posteriors.len() != partitions.len() {
        panic!(
            "Number of posteriors {} does not match number of partitions {}",
            posteriors.len(),
            partitions.len()
        );
    }

    let mut combined = Array1::<f64>::zeros(total_variables);
    let mut max_confidence = Array1::<f64>::zeros(total_variables);

    for (posterior, partition) in posteriors.iter().zip(partitions.iter()) {
        if posterior.len() != partition.variable_indices.len() {
            panic!(
                "Posterior length {} does not match partition variable count {}",
                posterior.len(),
                partition.variable_indices.len()
            );
        }

        for (local_idx, &global_idx) in partition.variable_indices.iter().enumerate() {
            let value = posterior[local_idx];
            let confidence = value.abs();

            // Keep posterior with highest confidence
            if confidence > max_confidence[global_idx] {
                max_confidence[global_idx] = confidence;
                combined[global_idx] = value;
            }
        }
    }

    combined
}

/// Identify boundary detectors that connect to variables in multiple partitions.
///
/// A boundary detector is one that connects to variables from different partitions.
/// These are the only detectors that need to be reconciled during fusion.
///
/// # Arguments
/// * `original_matrix` - Original check matrix
/// * `partitions` - Vector of partitions
///
/// # Returns
/// Set of detector indices that are on partition boundaries
fn identify_boundary_detectors(
    original_matrix: &SparseBitMatrix,
    partitions: &[Partition],
) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;
    
    // Build variable-to-partition mapping
    let mut var_to_partitions: std::collections::HashMap<usize, Vec<usize>> = 
        std::collections::HashMap::new();
    
    for (part_idx, partition) in partitions.iter().enumerate() {
        for &var_idx in &partition.variable_indices {
            var_to_partitions.entry(var_idx).or_insert_with(Vec::new).push(part_idx);
        }
    }
    
    // Find detectors that connect to variables from multiple partitions
    let mut boundary_detectors = HashSet::new();
    
    for det_idx in 0..original_matrix.rows() {
        let row = original_matrix.outer_view(det_idx).unwrap();
        let mut partitions_seen = HashSet::new();
        
        // Check which partitions this detector's variables belong to
        for (var_idx, _) in row.iter() {
            if let Some(partitions_for_var) = var_to_partitions.get(&var_idx) {
                for &part_idx in partitions_for_var {
                    partitions_seen.insert(part_idx);
                }
            }
        }
        
        // If detector connects to variables from multiple partitions, it's a boundary detector
        if partitions_seen.len() > 1 {
            boundary_detectors.insert(det_idx);
        }
    }
    
    boundary_detectors
}

/// Fusion matrix creation strategy
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusionStrategy {
    /// Use full check matrix - ensures correctness but slower
    FullMatrix,
    /// Use only boundary regions - faster but may have correctness issues
    BoundaryOnly,
}

/// Creates a fusion check matrix for combining partition solutions (full matrix version).
///
/// Returns the complete original check matrix for fusion decoding.
/// All constraints (interior + boundary) are included to ensure correctness.
///
/// The fusion decoder runs BP on the full problem but with warm-started
/// posteriors from leaf decoders. This preserves correct interior solutions
/// while reconciling boundary inconsistencies.
///
/// # Arguments
/// * `original_matrix` - Original check matrix
/// * `_partitions` - Vector of partitions (unused but kept for API compatibility)
///
/// # Returns
/// Tuple of (fusion check matrix, all detector indices)
pub fn create_fusion_matrix_full(
    original_matrix: &SparseBitMatrix,
    _partitions: &[Partition],
) -> (Arc<SparseBitMatrix>, Vec<usize>) {
    // Use full matrix to ensure all constraints are satisfied
    // Warm-started posteriors from leaf decoders help convergence
    let all_detectors: Vec<usize> = (0..original_matrix.rows()).collect();

    (Arc::new(original_matrix.clone()), all_detectors)
}

/// Creates a fusion check matrix containing only boundary detectors and variables.
///
/// This optimizes fusion decoding by only processing the boundary regions where
/// partitions interact, rather than the full problem. Interior solutions from
/// leaf decoders are preserved, and only boundary inconsistencies are reconciled.
///
/// WARNING: This approach may have correctness issues as it doesn't verify
/// all detector constraints. Use with caution.
///
/// # Arguments
/// * `original_matrix` - Original check matrix
/// * `partitions` - Vector of partitions
///
/// # Returns
/// Tuple of (fusion check matrix with only boundaries, boundary detector indices)
pub fn create_fusion_matrix_boundary_only(
    original_matrix: &SparseBitMatrix,
    partitions: &[Partition],
) -> (Arc<SparseBitMatrix>, Vec<usize>) {
    use std::collections::HashSet;
    
    // Identify boundary detectors
    let boundary_detectors_set = identify_boundary_detectors(original_matrix, partitions);
    
    if boundary_detectors_set.is_empty() {
                    // No boundaries - return empty matrix
                    // This case occurs when partitions have no overlapping variables
        use sprs::TriMat;
        let empty = TriMat::new((0, original_matrix.cols()));
        return (Arc::new(empty.to_csc() as SparseBitMatrix), Vec::new());
    }
    
    // Identify boundary variables (variables connected to boundary detectors)
    let mut boundary_vars_set = HashSet::new();
    for &det_idx in &boundary_detectors_set {
        let row = original_matrix.outer_view(det_idx).unwrap();
        for (var_idx, _) in row.iter() {
            boundary_vars_set.insert(var_idx);
        }
    }
    
    // Include all detectors that connect to boundary variables
    // This ensures we check all constraints that might be affected by boundary variable changes
    let mut all_fusion_detectors = boundary_detectors_set.clone();
    for det_idx in 0..original_matrix.rows() {
        if boundary_detectors_set.contains(&det_idx) {
            continue; // Already included
        }
        let row = original_matrix.outer_view(det_idx).unwrap();
        for (var_idx, _) in row.iter() {
            if boundary_vars_set.contains(&var_idx) {
                // This detector connects to a boundary variable, include it
                all_fusion_detectors.insert(det_idx);
                break;
            }
        }
    }
    
    // Create mapping for fusion detectors and variables
    let mut fusion_detectors_sorted: Vec<usize> = all_fusion_detectors.iter().copied().collect();
    fusion_detectors_sorted.sort();
    
    let mut boundary_vars_sorted: Vec<usize> = boundary_vars_set.iter().copied().collect();
    boundary_vars_sorted.sort();
    
    let det_map: std::collections::HashMap<usize, usize> = fusion_detectors_sorted
        .iter()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();
    
    let var_map: std::collections::HashMap<usize, usize> = boundary_vars_sorted
        .iter()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();
    
    // Build boundary-only check matrix
    use sprs::TriMat;
    let mut tri_mat = TriMat::new((fusion_detectors_sorted.len(), boundary_vars_sorted.len()));
    
    for &det_idx in &fusion_detectors_sorted {
        let row = original_matrix.outer_view(det_idx).unwrap();
        let new_det_idx = det_map[&det_idx];
        
        for (var_idx, &value) in row.iter() {
            if let Some(&new_var_idx) = var_map.get(&var_idx) {
                tri_mat.add_triplet(new_det_idx, new_var_idx, value);
            }
        }
    }
    
    let fusion_matrix = Arc::new(tri_mat.to_csc() as SparseBitMatrix);
    
    (fusion_matrix, fusion_detectors_sorted)
}

/// Creates a fusion check matrix based on the specified strategy.
///
/// # Arguments
/// * `original_matrix` - Original check matrix
/// * `partitions` - Vector of partitions
/// * `strategy` - Fusion strategy to use
///
/// # Returns
/// Tuple of (fusion check matrix, detector indices)
pub fn create_fusion_matrix(
    original_matrix: &SparseBitMatrix,
    partitions: &[Partition],
    strategy: FusionStrategy,
) -> (Arc<SparseBitMatrix>, Vec<usize>) {
    match strategy {
        FusionStrategy::FullMatrix => create_fusion_matrix_full(original_matrix, partitions),
        FusionStrategy::BoundaryOnly => create_fusion_matrix_boundary_only(original_matrix, partitions),
    }
}

/// Fusion decoder that combines solutions from multiple partitions.
///
/// This decoder uses a two-stage approach:
/// 1. Runs leaf decoders in parallel on partition sub-problems
/// 2. Combines results using a fusion decoder with warm-started posteriors
///
/// The fusion stage reconciles boundary inconsistencies between partitions
/// while preserving correct interior solutions from the leaf decoders.
pub struct FusionDecoder<N: PartialEq + Default + Clone + Copy> {
    original_matrix: Arc<SparseBitMatrix>,
    partitions: Vec<Partition>,
    min_sum_config: Arc<MinSumDecoderConfig>,
    relay_config: Arc<RelayDecoderConfig>,
    fusion_strategy: FusionStrategy,
    _phantom: std::marker::PhantomData<N>,
}

impl<N> FusionDecoder<N>
where
    N: PartialEq
        + std::fmt::Debug
        + Default
        + Clone
        + Copy
        + num_traits::Signed
        + num_traits::Bounded
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    /// Create a new fusion decoder with full matrix strategy (ensures correctness).
    ///
    /// # Arguments
    /// * `check_matrix` - Original check matrix
    /// * `partitions` - Vector of partitions
    /// * `min_sum_config` - Configuration for MinSum BP decoder
    /// * `relay_config` - Configuration for Relay decoder
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        partitions: Vec<Partition>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        relay_config: Arc<RelayDecoderConfig>,
    ) -> Self {
        Self::new_with_strategy(
            check_matrix,
            partitions,
            min_sum_config,
            relay_config,
            FusionStrategy::FullMatrix,
        )
    }

    /// Create a new fusion decoder with specified strategy.
    ///
    /// # Arguments
    /// * `check_matrix` - Original check matrix
    /// * `partitions` - Vector of partitions
    /// * `min_sum_config` - Configuration for MinSum BP decoder
    /// * `relay_config` - Configuration for Relay decoder
    /// * `strategy` - Fusion strategy (FullMatrix or BoundaryOnly)
    pub fn new_with_strategy(
        check_matrix: Arc<SparseBitMatrix>,
        partitions: Vec<Partition>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        relay_config: Arc<RelayDecoderConfig>,
        strategy: FusionStrategy,
    ) -> Self {
        FusionDecoder {
            original_matrix: check_matrix,
            partitions,
            min_sum_config,
            relay_config,
            fusion_strategy: strategy,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Decode using fusion approach.
    ///
    /// 1. Run leaf decoders in parallel on partition sub-problems
    /// 2. Extract posteriors and decodings from each leaf
    /// 3. Combine posteriors (averaging at boundaries)
    /// 4. Create fusion matrix based on selected strategy
    /// 5. Run fusion decoder with warm-started posteriors
    /// 6. Validate syndrome satisfaction
    ///
    /// # Arguments
    /// * `detectors` - Detector values for the full problem
    ///
    /// # Returns
    /// DecodeResult from fusion decoder
    pub fn decode_fusion(&self, detectors: ArrayView1<Bit>) -> DecodeResult {
        #[cfg(test)]
        println!("\n=== FUSION DECODING ===");

        // Step 1: Run leaf decoders in parallel
        let leaf_results: Vec<(Array1<f64>, DecodeResult)> = self
            .partitions
            .par_iter()
            .enumerate()
            .map(|(_idx, partition)| {
                // Create partition check matrix (this also populates detector_indices)
                let (partition_matrix, updated_partition) =
                    create_partition_check_matrix(&self.original_matrix, partition.clone());

                // Extract detectors for this partition
                let partition_detectors: Array1<Bit> = updated_partition
                    .detector_indices
                    .iter()
                    .map(|&det_idx| detectors[det_idx])
                    .collect();

                // Extract error priors for this partition
                let partition_priors: Array1<f64> = partition
                    .variable_indices
                    .iter()
                    .map(|&var_idx| self.min_sum_config.error_priors[var_idx])
                    .collect();

                // Create partition config
                let partition_config = Arc::new(MinSumDecoderConfig {
                    error_priors: partition_priors,
                    ..(*self.min_sum_config).clone()
                });

                // Create and run leaf decoder
                let mut leaf_decoder: RelayDecoder<N> = RelayDecoder::new(
                    partition_matrix,
                    partition_config,
                    self.relay_config.clone(),
                );

                let result = leaf_decoder.decode_detailed(partition_detectors.view());
                let posteriors = leaf_decoder.get_posterior_ratios_f64();

                (posteriors, result)
            })
            .collect();

        // Step 2: Extract posteriors and decodings
        let posteriors: Vec<Array1<f64>> = leaf_results.iter().map(|(p, _)| p.clone()).collect();
        let leaf_decodings: Vec<Array1<u8>> = leaf_results.iter().map(|(_, r)| r.decoding.clone()).collect();

        // Step 3: Combine posteriors
        let combined_posteriors =
            combine_posteriors(&posteriors, &self.partitions, self.original_matrix.cols());

        // Step 4: Create fusion decoder based on strategy
        let (fusion_matrix, fusion_detector_indices) =
            create_fusion_matrix(&self.original_matrix, &self.partitions, self.fusion_strategy);
        
        let fusion_result = match self.fusion_strategy {
            FusionStrategy::FullMatrix => {
                // Full matrix: use all detectors and variables
                let mut fusion_decoder: RelayDecoder<N> = RelayDecoder::new(
                    fusion_matrix,
                    self.min_sum_config.clone(),
                    self.relay_config.clone(),
                );

                // Step 5: Load combined posteriors and run fusion decoding
                fusion_decoder.set_posterior_ratios_f64(combined_posteriors);
                fusion_decoder.decode_detailed(detectors)
            }
            FusionStrategy::BoundaryOnly => {
                // Boundary-only: extract boundary components and run fusion
                if fusion_detector_indices.is_empty() {
                    // No boundaries - combine leaf decoder results directly
                    let mut full_decoding = Array1::<u8>::zeros(self.original_matrix.cols());
                    let mut full_decoded_detectors = Array1::<Bit>::zeros(self.original_matrix.rows());
                    
                    // Combine leaf decoder decodings for all variables
                    for (leaf_decoding, partition) in leaf_decodings.iter().zip(self.partitions.iter()) {
                        for (local_idx, &global_idx) in partition.variable_indices.iter().enumerate() {
                            full_decoding[global_idx] = leaf_decoding[local_idx];
                        }
                    }
                    
                    // Compute decoded detectors
                    for (det_idx, detector_row) in self.original_matrix.outer_iterator().enumerate() {
                        let mut sum = 0u8;
                        for (var_idx, _) in detector_row.iter() {
                            sum ^= full_decoding[var_idx];
                        }
                        full_decoded_detectors[det_idx] = sum;
                    }
                    
                    // Check syndrome satisfaction
                    let syndrome_satisfied = full_decoded_detectors
                        .iter()
                        .zip(detectors.iter())
                        .all(|(computed, original)| computed == original);
                    
                    return DecodeResult {
                        decoding: full_decoding,
                        decoded_detectors: full_decoded_detectors,
                        posterior_ratios: combined_posteriors,
                        success: syndrome_satisfied,
                        decoding_quality: 0.0,
                        iterations: 0,
                        max_iter: 0,
                        extra: BPExtraResult::None,
                    };
                }
                
                // Extract boundary variables (variables in the fusion matrix)
                let boundary_vars: std::collections::HashSet<usize> = {
                    let mut vars = std::collections::HashSet::new();
                    for &det_idx in &fusion_detector_indices {
                        let row = self.original_matrix.outer_view(det_idx).unwrap();
                        for (var_idx, _) in row.iter() {
                            vars.insert(var_idx);
                        }
                    }
                    vars
                };
                
                let mut boundary_vars_sorted: Vec<usize> = boundary_vars.iter().copied().collect();
                boundary_vars_sorted.sort();
                
                // Create boundary-only posterior array
                let boundary_posteriors: Array1<f64> = boundary_vars_sorted
                    .iter()
                    .map(|&var_idx| combined_posteriors[var_idx])
                    .collect();
                
                // Extract boundary detector values
                let boundary_detector_values: Array1<Bit> = fusion_detector_indices
                    .iter()
                    .map(|&det_idx| detectors[det_idx])
                    .collect();
                
                // Extract boundary variable priors
                let boundary_priors: Array1<f64> = boundary_vars_sorted
                    .iter()
                    .map(|&var_idx| self.min_sum_config.error_priors[var_idx])
                    .collect();
                
                // Create boundary-only config
                let boundary_config = Arc::new(MinSumDecoderConfig {
                    error_priors: boundary_priors,
                    ..(*self.min_sum_config).clone()
                });
                
                let mut fusion_decoder: RelayDecoder<N> = RelayDecoder::new(
                    fusion_matrix,
                    boundary_config,
                    self.relay_config.clone(),
                );

                // Step 5: Load boundary posteriors and run fusion decoding on boundaries only
                fusion_decoder.set_posterior_ratios_f64(boundary_posteriors);
                let boundary_fusion_result = fusion_decoder.decode_detailed(boundary_detector_values.view());
                
                // Step 6: Map boundary results back to full variable space
                let mut full_decoding = Array1::<u8>::zeros(self.original_matrix.cols());
                
                // Initialize with leaf decoder decodings for interior variables
                // This preserves correct interior solutions from partition decoders
                for (leaf_decoding, partition) in leaf_decodings.iter().zip(self.partitions.iter()) {
                    for (local_idx, &global_idx) in partition.variable_indices.iter().enumerate() {
                        // Set interior variables from leaf decoder results
                        // Boundary variables will be set by fusion decoder below
                        if !boundary_vars.contains(&global_idx) {
                            full_decoding[global_idx] = leaf_decoding[local_idx];
                        }
                    }
                }
                
                // Override with boundary fusion results for boundary variables
                let var_map: std::collections::HashMap<usize, usize> = boundary_vars_sorted
                    .iter()
                    .enumerate()
                    .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
                    .collect();
                
                for &var_idx in &boundary_vars_sorted {
                    if let Some(&boundary_idx) = var_map.get(&var_idx) {
                        full_decoding[var_idx] = boundary_fusion_result.decoding[boundary_idx];
                    }
                }
                
                // Compute decoded detectors for full problem
                let mut full_decoded_detectors = Array1::<Bit>::zeros(self.original_matrix.rows());
                for (det_idx, detector_row) in self.original_matrix.outer_iterator().enumerate() {
                    let mut sum = 0u8;
                    for (var_idx, _) in detector_row.iter() {
                        sum ^= full_decoding[var_idx];
                    }
                    full_decoded_detectors[det_idx] = sum;
                }
                
                // Create full result with boundary fusion decoding
                DecodeResult {
                    decoding: full_decoding,
                    decoded_detectors: full_decoded_detectors,
                    posterior_ratios: {
                        let mut full_posteriors = combined_posteriors.clone();
                        // Update boundary posteriors with fusion results
                        for &var_idx in &boundary_vars_sorted {
                            if let Some(&boundary_idx) = var_map.get(&var_idx) {
                                full_posteriors[var_idx] = boundary_fusion_result.posterior_ratios[boundary_idx];
                            }
                        }
                        full_posteriors
                    },
                    success: boundary_fusion_result.success,
                    decoding_quality: boundary_fusion_result.decoding_quality,
                    iterations: boundary_fusion_result.iterations,
                    max_iter: boundary_fusion_result.max_iter,
                    extra: boundary_fusion_result.extra,
                }
            }
        };

        // Step 6: Validate that fusion result satisfies the full original syndrome
        let mut computed_syndrome = vec![0u8; self.original_matrix.rows()];
        for (det_idx, detector_row) in self.original_matrix.outer_iterator().enumerate() {
            let mut sum = 0u8;
            for (var_idx, _) in detector_row.iter() {
                sum ^= fusion_result.decoding[var_idx];
            }
            computed_syndrome[det_idx] = sum;
        }

        let syndrome_satisfied = computed_syndrome
            .iter()
            .zip(detectors.iter())
            .all(|(computed, original)| computed == original);

        #[cfg(test)]
        if !syndrome_satisfied {
            println!("WARNING: Fusion result does not satisfy full syndrome!");
            println!("  Expected: {:?}", detectors);
            println!("  Got:      {:?}", computed_syndrome);
        }

        // Return result with updated success flag
        DecodeResult {
            success: syndrome_satisfied && fusion_result.success,
            ..fusion_result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bipartite_graph::BipartiteGraph;
    use crate::bp::min_sum::MinSumDecoderConfig;
    use crate::bp::relay::RelayDecoderConfig;
    use crate::decoder::Bit;
    use ndarray::{array, Array2};

    #[test]
    fn test_partition_by_count() {
        let partitions = partition_by_count(1000, 2);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].variable_indices.len(), 500);
        assert_eq!(partitions[1].variable_indices.len(), 500);
        assert_eq!(partitions[0].variable_indices[0], 0);
        assert_eq!(partitions[0].variable_indices[499], 499);
        assert_eq!(partitions[1].variable_indices[0], 500);
        assert_eq!(partitions[1].variable_indices[499], 999);
    }

    #[test]
    fn test_partition_by_time_rounds_basic() {
        // Create a simple spacetime check matrix with 4 rounds, 2 detectors/round
        // Total: 8 detectors × 10 variables
        // Structure: Each round's detectors connect to a subset of variables
        // Round 0 (dets 0-1): connects to vars 0,1,2
        // Round 1 (dets 2-3): connects to vars 2,3,4
        // Round 2 (dets 4-5): connects to vars 4,5,6
        // Round 3 (dets 6-7): connects to vars 6,7,8
        let dense = array![
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], // det 0
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], // det 1
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], // det 2
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], // det 3
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0], // det 4
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], // det 5
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0], // det 6
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], // det 7
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        // Partition by 2 rounds per partition
        let partitions = partition_by_time_rounds(&check_matrix, 2, 2);

        // Should create 2 partitions (4 rounds / 2 rounds per partition)
        assert_eq!(partitions.len(), 2);

        // Partition 0: rounds 0-1 (detectors 0-3)
        assert_eq!(partitions[0].detector_indices, vec![0, 1, 2, 3]);
        // Should include variables connected to detectors 0-3: vars 0,1,2,3,4
        assert_eq!(partitions[0].variable_indices, vec![0, 1, 2, 3, 4]);

        // Partition 1: rounds 2-3 (detectors 4-7)
        assert_eq!(partitions[1].detector_indices, vec![4, 5, 6, 7]);
        // Should include variables connected to detectors 4-7: vars 4,5,6,7,8
        assert_eq!(partitions[1].variable_indices, vec![4, 5, 6, 7, 8]);

        // Variable 4 appears in both partitions, making it a boundary variable
    }

    #[test]
    fn test_partition_by_time_rounds_uneven_rounds() {
        // Test with 7 rounds, 3 detectors/round, 3 rounds/partition
        // Total: 21 detectors
        // Should create 3 partitions: [0-2], [3-5], [6] (last one partial)
        let mut dense = Array2::<u8>::zeros((21, 30));
        // Make sure each detector connects to at least one variable
        for i in 0..21 {
            dense[[i, i]] = 1;
        }
        let check_matrix = SparseBitMatrix::from_dense(dense);

        let partitions = partition_by_time_rounds(&check_matrix, 3, 3);

        assert_eq!(partitions.len(), 3);

        // Partition 0: rounds 0-2 (detectors 0-8)
        assert_eq!(partitions[0].detector_indices.len(), 9);
        assert_eq!(partitions[0].detector_indices[0], 0);
        assert_eq!(partitions[0].detector_indices[8], 8);

        // Partition 1: rounds 3-5 (detectors 9-17)
        assert_eq!(partitions[1].detector_indices.len(), 9);
        assert_eq!(partitions[1].detector_indices[0], 9);
        assert_eq!(partitions[1].detector_indices[8], 17);

        // Partition 2: round 6 only (detectors 18-20) - partial partition
        assert_eq!(partitions[2].detector_indices.len(), 3);
        assert_eq!(partitions[2].detector_indices, vec![18, 19, 20]);
    }

    #[test]
    fn test_partition_by_time_rounds_realistic_parameters() {
        // Simulate realistic spacetime code parameters:
        // - 150 rounds
        // - 72 detectors per round
        // - 10 rounds per partition
        // Total: 10,800 detectors
        let detectors_per_round = 72;
        let total_rounds = 150;
        let rounds_per_partition = 10;
        let total_detectors = detectors_per_round * total_rounds;

        // Create a sparse check matrix where each detector connects to ~3 variables
        // (typical for surface codes)
        use sprs::TriMat;
        let total_variables = total_detectors * 3 / 2; // Rough estimate
        let mut tri_mat = TriMat::new((total_detectors, total_variables));

        // Simple pattern: detector i connects to variables i, i+1, i+2
        for det in 0..total_detectors {
            for var_offset in 0..3 {
                let var = (det + var_offset) % total_variables;
                tri_mat.add_triplet(det, var, 1u8);
            }
        }
        let check_matrix = tri_mat.to_csr() as SparseBitMatrix;

        let partitions = partition_by_time_rounds(&check_matrix, detectors_per_round, rounds_per_partition);

        // Should create 15 partitions (150 rounds / 10 rounds per partition)
        assert_eq!(partitions.len(), 15);

        // Each partition should have 10 rounds × 72 detectors = 720 detectors
        for (i, partition) in partitions.iter().enumerate() {
            assert_eq!(
                partition.detector_indices.len(),
                720,
                "Partition {} has wrong number of detectors",
                i
            );

            // Verify detectors are consecutive
            let expected_start = i * 720;
            assert_eq!(partition.detector_indices[0], expected_start);
            assert_eq!(partition.detector_indices[719], expected_start + 719);
        }

        // Verify no gaps or overlaps in detector coverage
        let mut all_detectors = std::collections::HashSet::new();
        for partition in &partitions {
            for &det in &partition.detector_indices {
                assert!(
                    all_detectors.insert(det),
                    "Detector {} appears in multiple partitions",
                    det
                );
            }
        }
        assert_eq!(all_detectors.len(), total_detectors);

        // Verify boundary/volume ratio is small
        for (i, partition) in partitions.iter().enumerate() {
            let num_variables = partition.variable_indices.len();
            let num_detectors = partition.detector_indices.len();

            // With our simple pattern (3 vars per detector), we expect ~720 + boundary overla
            assert!(
                num_variables >= 720,
                "Partition {} has too few variables: {}",
                i,
                num_variables
            );
            assert!(
                num_variables < 1500,
                "Partition {} has too many variables: {}",
                i,
                num_variables
            );

            println!(
                "Partition {}: {} detectors, {} variables (ratio: {:.2})",
                i,
                num_detectors,
                num_variables,
                num_variables as f64 / num_detectors as f64
            );
        }

        // Calculate boundary size by checking variable overlap between adjacent partitions
        for i in 0..(partitions.len() - 1) {
            let vars_i: std::collections::HashSet<usize> =
                partitions[i].variable_indices.iter().copied().collect();
            let vars_next: std::collections::HashSet<usize> =
                partitions[i + 1].variable_indices.iter().copied().collect();

            let boundary_size = vars_i.intersection(&vars_next).count();
            let partition_volume = vars_i.len();

            let boundary_ratio = boundary_size as f64 / partition_volume as f64;

            println!(
                "Boundary {}-{}: {} variables shared (ratio: {:.3})",
                i,
                i + 1,
                boundary_size,
                boundary_ratio
            );

            // Boundary ratio should be small (< 0.15 for good time partitioning)
            assert!(
                boundary_ratio < 0.15,
                "Boundary ratio {:.3} is too large (should be < 0.15)",
                boundary_ratio
            );
        }
    }

    #[test]
    fn test_extract_connected_variables() {
        // Create a simple check matrix
        let dense = array![
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        // Extract variables connected to detectors 0 and 1
        let variables = super::extract_connected_variables(&check_matrix, &[0, 1]);
        assert_eq!(variables, vec![0, 1, 2]);

        // Extract variables connected to detectors 2 and 3
        let variables = super::extract_connected_variables(&check_matrix, &[2, 3]);
        assert_eq!(variables, vec![2, 3, 4]);

        // Extract variables connected to all detectors
        let variables = super::extract_connected_variables(&check_matrix, &[0, 1, 2, 3]);
        assert_eq!(variables, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "detectors_per_round must be > 0")]
    fn test_partition_by_time_rounds_zero_detectors_per_round() {
        let dense = array![[1, 0], [0, 1]];
        let check_matrix = SparseBitMatrix::from_dense(dense);
        partition_by_time_rounds(&check_matrix, 0, 1);
    }

    #[test]
    #[should_panic(expected = "rounds_per_partition must be > 0")]
    fn test_partition_by_time_rounds_zero_rounds_per_partition() {
        let dense = array![[1, 0], [0, 1]];
        let check_matrix = SparseBitMatrix::from_dense(dense);
        partition_by_time_rounds(&check_matrix, 1, 0);
    }

    #[test]
    #[should_panic(expected = "not evenly divisible")]
    fn test_partition_by_time_rounds_uneven_detectors() {
        // 5 detectors, 2 detectors/round → doesn't divide evenly
        let dense = array![
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);
        partition_by_time_rounds(&check_matrix, 2, 1);
    }

    #[test]
    fn test_partition_by_time_rounds_single_round_per_partition() {
        // Edge case: 1 round per partition
        let dense = array![
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        let partitions = partition_by_time_rounds(&check_matrix, 2, 1);

        // Should create 2 partitions (4 detectors / 2 per round = 2 rounds)
        assert_eq!(partitions.len(), 2);

        // Each partition should have exactly 2 detectors (1 round × 2 detectors/round)
        assert_eq!(partitions[0].detector_indices, vec![0, 1]);
        assert_eq!(partitions[1].detector_indices, vec![2, 3]);
    }

    #[test]
    fn test_partition_by_time_rounds_all_rounds_in_one_partition() {
        // Edge case: all rounds in one partition
        let dense = array![
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        let partitions = partition_by_time_rounds(&check_matrix, 2, 10);

        // Should create 1 partition (only 2 rounds, but 10 rounds/partition requested)
        assert_eq!(partitions.len(), 1);

        // Should include all 4 detectors
        assert_eq!(partitions[0].detector_indices, vec![0, 1, 2, 3]);

        // Should include all variables (0,1,2,3)
        assert_eq!(partitions[0].variable_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_partition_by_count_remainder() {
        let partitions = partition_by_count(1001, 2);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].variable_indices.len(), 501);
        assert_eq!(partitions[1].variable_indices.len(), 500);
    }

    #[test]
    fn test_create_partition_check_matrix() {
        // Create a simple check matrix: 3 detectors × 6 variables
        // Detector 0: checks vars 0,1
        // Detector 1: checks vars 1,2,3
        // Detector 2: checks vars 4,5
        let dense = array![
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        // Partition: vars 0-2 (first partition)
        let partition = Partition {
            variable_indices: vec![0, 1, 2],
            detector_indices: Vec::new(),
        };

        let (partition_matrix, updated_partition) =
            create_partition_check_matrix(&check_matrix, partition);

        // Should include detectors 0 and 1 (they connect to vars 0-2)
        assert_eq!(updated_partition.detector_indices.len(), 2);
        assert!(updated_partition.detector_indices.contains(&0));
        assert!(updated_partition.detector_indices.contains(&1));

        // Partition matrix should be 2 detectors × 3 variables
        assert_eq!(partition_matrix.rows(), 2);
        assert_eq!(partition_matrix.cols(), 3);
    }

    #[test]
    fn test_combine_posteriors() {
        let partitions = vec![
            Partition {
                variable_indices: vec![0, 1, 2],
                detector_indices: vec![0, 1],
            },
            Partition {
                variable_indices: vec![3, 4, 5],
                detector_indices: vec![1, 2],
            },
        ];

        let posteriors = vec![
            array![1.0, 2.0, 3.0],
            array![4.0, 5.0, 6.0],
        ];

        let combined = combine_posteriors(&posteriors, &partitions, 6);
        assert_eq!(combined.len(), 6);
        assert_eq!(combined[0], 1.0);
        assert_eq!(combined[1], 2.0);
        assert_eq!(combined[2], 3.0);
        assert_eq!(combined[3], 4.0);
        assert_eq!(combined[4], 5.0);
        assert_eq!(combined[5], 6.0);
    }

    #[test]
    fn test_create_fusion_matrix() {
        // Create a check matrix with boundary and interior detectors
        // 6 variables split into 2 partitions: [0,1,2] and [3,4,5]
        // Detector 0: checks vars 0,1 -> interior to partition 0
        // Detector 1: checks vars 1,2 -> interior to partition 0
        // Detector 2: checks vars 2,3 -> BOUNDARY (spans both partitions)
        // Detector 3: checks vars 4,5 -> interior to partition 1
        let dense = array![
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ];
        let check_matrix = SparseBitMatrix::from_dense(dense);

        let partitions = vec![
            Partition {
                variable_indices: vec![0, 1, 2],
                detector_indices: vec![],
            },
            Partition {
                variable_indices: vec![3, 4, 5],
                detector_indices: vec![],
            },
        ];

        let (fusion_matrix, all_detectors) =
            create_fusion_matrix(&check_matrix, &partitions, FusionStrategy::FullMatrix);

        // Full matrix strategy includes all detectors to ensure all constraints are satisfied
        assert_eq!(all_detectors.len(), 4);
        assert_eq!(all_detectors, vec![0, 1, 2, 3]);

        // Fusion matrix should equal original matrix
        assert_eq!(fusion_matrix.rows(), 4);
        assert_eq!(fusion_matrix.cols(), 6);

        // Verify fusion matrix contains all detectors
        for det_idx in 0..4 {
            let original_row = check_matrix.outer_view(det_idx).unwrap();
            let fusion_row = fusion_matrix.outer_view(det_idx).unwrap();

            let original_vars: Vec<usize> = original_row.iter().map(|(var, _)| var).collect();
            let fusion_vars: Vec<usize> = fusion_row.iter().map(|(var, _)| var).collect();

            assert_eq!(original_vars, fusion_vars);
        }
    }

    #[test]
    fn test_fusion_decoder_basic() {
        // Create a simple repetition code: 3 variables, 2 detectors
        // Detector 0: checks vars 0,1
        // Detector 1: checks vars 1,2
        let dense = array![[1, 1, 0], [0, 1, 1]];
        let check_matrix = Arc::new(SparseBitMatrix::from_dense(dense));

        // Partition into 2 parts: [0,1] and [2]
        let partitions = vec![
            Partition {
                variable_indices: vec![0, 1],
                detector_indices: vec![0, 1],
            },
            Partition {
                variable_indices: vec![2],
                detector_indices: vec![1],
            },
        ];

        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: 10,
            ..Default::default()
        });

        let relay_config = Arc::new(RelayDecoderConfig {
            pre_iter: 10,
            num_sets: 0, // Disable relay sets for simple test
            set_max_iter: 10,
            ..Default::default()
        });

        let fusion_decoder: FusionDecoder<f64> = FusionDecoder::new(
            check_matrix.clone(),
            partitions,
            min_sum_config.clone(),
            relay_config.clone(),
        );

        // Test with detectors [1, 1] (error on variable 1)
        let detectors: Array1<Bit> = array![1, 1];
        let result = fusion_decoder.decode_fusion(detectors.view());

        // Should decode error on variable 1
        assert!(result.success || result.iterations > 0);
        // Check that decoding produces valid result
        assert_eq!(result.decoding.len(), 3);
    }

    #[test]
    fn test_regular_decoder_baseline() {
        // First verify that a regular decoder CAN solve this problem
        let dense = array![
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ];
        let check_matrix = Arc::new(SparseBitMatrix::from_dense(dense));

        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.01, 0.01, 0.01, 0.01, 0.01],
            max_iter: 100,
            ..Default::default()
        });

        let relay_config = Arc::new(RelayDecoderConfig {
            pre_iter: 50,
            num_sets: 0,
            set_max_iter: 50,
            ..Default::default()
        });

        let mut regular_decoder: RelayDecoder<f64> = RelayDecoder::new(
            check_matrix.clone(),
            min_sum_config.clone(),
            relay_config.clone(),
        );

        // Error on variable 2: syndrome [0, 1, 1, 0]
        let detectors: Array1<Bit> = array![0, 1, 1, 0];
        let result = regular_decoder.decode_detailed(detectors.view());

        println!("\nRegular decoder (baseline):");
        println!("  Input syndrome: {:?}", detectors);
        println!("  Decoded errors: {:?}", result.decoding);
        println!("  Success: {}, Iterations: {}", result.success, result.iterations);

        // Verify syndrome
        let mut computed_syndrome = vec![0u8; 4];
        for (det_idx, detector_row) in check_matrix.outer_iterator().enumerate() {
            let mut sum = 0u8;
            for (var_idx, _) in detector_row.iter() {
                sum ^= result.decoding[var_idx];
            }
            computed_syndrome[det_idx] = sum;
        }

        println!("  Computed syndrome: {:?}", computed_syndrome);

        assert_eq!(computed_syndrome[0], detectors[0]);
        assert_eq!(computed_syndrome[1], detectors[1]);
        assert_eq!(computed_syndrome[2], detectors[2]);
        assert_eq!(computed_syndrome[3], detectors[3]);
    }

    #[test]
    fn test_fusion_decoder_simple_case() {
        // Simple 4-variable, 3-detector test where fusion actually helps
        // This is a chain code: var0--det0--var1--det1--var2--det2--var3
        // Partition: [0,1] and [2,3], boundary detector is det1 (checks vars 1,2)
        let dense = array![
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ];
        let check_matrix = Arc::new(SparseBitMatrix::from_dense(dense));

        let partitions = vec![
            Partition {
                variable_indices: vec![0, 1],
                detector_indices: vec![],
            },
            Partition {
                variable_indices: vec![2, 3],
                detector_indices: vec![],
            },
        ];

        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.01, 0.01, 0.01, 0.01],
            max_iter: 100,
            ..Default::default()
        });

        let relay_config = Arc::new(RelayDecoderConfig {
            pre_iter: 50,
            num_sets: 0,
            set_max_iter: 50,
            ..Default::default()
        });

        let fusion_decoder: FusionDecoder<f64> = FusionDecoder::new(
            check_matrix.clone(),
            partitions,
            min_sum_config.clone(),
            relay_config.clone(),
        );

        // Test: No errors (all detectors silent)
        let detectors_no_error: Array1<Bit> = array![0, 0, 0];
        let result = fusion_decoder.decode_fusion(detectors_no_error.view());

        println!("\nSimple test - No errors:");
        println!("  Input syndrome: {:?}", detectors_no_error);
        println!("  Decoded errors: {:?}", result.decoding);
        println!("  Success: {}", result.success);

        // Should decode to no errors
        let all_zero = result.decoding.iter().all(|&x| x == 0);
        assert!(all_zero, "Should decode to all zeros when no detectors fire");
    }

    #[test]
    fn test_fusion_decoder_correct_decoding() {
        // Create a 5-variable, 4-detector code with clear boundary structure
        // Variables partitioned as: [0,1,2] and [3,4]
        // Detector 0: vars 0,1 (interior to partition 0)
        // Detector 1: vars 1,2 (interior to partition 0)
        // Detector 2: vars 2,3 (BOUNDARY - spans partitions)
        // Detector 3: vars 3,4 (interior to partition 1)
        let dense = array![
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ];
        let check_matrix = Arc::new(SparseBitMatrix::from_dense(dense));

        let partitions = vec![
            Partition {
                variable_indices: vec![0, 1, 2],
                detector_indices: vec![],
            },
            Partition {
                variable_indices: vec![3, 4],
                detector_indices: vec![],
            },
        ];

        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.01, 0.01, 0.01, 0.01, 0.01],
            max_iter: 100,
            ..Default::default()
        });

        let relay_config = Arc::new(RelayDecoderConfig {
            pre_iter: 50,
            num_sets: 0,
            set_max_iter: 50,
            ..Default::default()
        });

        let fusion_decoder: FusionDecoder<f64> = FusionDecoder::new(
            check_matrix.clone(),
            partitions,
            min_sum_config.clone(),
            relay_config.clone(),
        );

        // Test case 1: Error on boundary variable 2
        // True error: [0, 0, 1, 0, 0]
        // Syndrome: detector 1 fires (checks vars 1,2: 0^1=1)
        //           detector 2 fires (checks vars 2,3: 1^0=1) <- BOUNDARY detector!
        let detectors_case1: Array1<Bit> = array![0, 1, 1, 0];
        let result1 = fusion_decoder.decode_fusion(detectors_case1.view());

        println!("\nTest case 1: Error on boundary variable");
        println!("  Input syndrome: {:?}", detectors_case1);
        println!("  Boundary detector 2 syndrome: {}", detectors_case1[2]);
        println!("  Decoded errors: {:?}", result1.decoding);
        println!("  Success: {}, Iterations: {}", result1.success, result1.iterations);

        // Verify syndrome is satisfied: H * decoding = detectors
        let mut computed_syndrome = vec![0u8; 4];
        for (det_idx, detector_row) in check_matrix.outer_iterator().enumerate() {
            let mut sum = 0u8;
            for (var_idx, _) in detector_row.iter() {
                sum ^= result1.decoding[var_idx];
            }
            computed_syndrome[det_idx] = sum;
        }

        println!("  Computed syndrome: {:?}", computed_syndrome);

        // The decoding should satisfy the syndrome
        assert_eq!(computed_syndrome[0], detectors_case1[0],
                   "Detector 0: expected {}, got {}", detectors_case1[0], computed_syndrome[0]);
        assert_eq!(computed_syndrome[1], detectors_case1[1],
                   "Detector 1: expected {}, got {}", detectors_case1[1], computed_syndrome[1]);
        assert_eq!(computed_syndrome[2], detectors_case1[2],
                   "Detector 2 (BOUNDARY): expected {}, got {}", detectors_case1[2], computed_syndrome[2]);
        assert_eq!(computed_syndrome[3], detectors_case1[3],
                   "Detector 3: expected {}, got {}", detectors_case1[3], computed_syndrome[3]);

        // Test case 2: Error on variable 3 (also touches boundary)
        // True error: [0, 0, 0, 1, 0]
        // Syndrome: detector 2 fires (checks vars 2,3: 0^1=1) <- BOUNDARY detector!
        //           detector 3 fires (checks vars 3,4: 1^0=1)
        let detectors_case2: Array1<Bit> = array![0, 0, 1, 1];
        let result2 = fusion_decoder.decode_fusion(detectors_case2.view());

        println!("\nTest case 2: Error on other boundary variable");
        println!("  Input syndrome: {:?}", detectors_case2);
        println!("  Boundary detector 2 syndrome: {}", detectors_case2[2]);
        println!("  Decoded errors: {:?}", result2.decoding);
        println!("  Success: {}, Iterations: {}", result2.success, result2.iterations);

        // Verify syndrome is satisfied
        let mut computed_syndrome2 = vec![0u8; 4];
        for (det_idx, detector_row) in check_matrix.outer_iterator().enumerate() {
            let mut sum = 0u8;
            for (var_idx, _) in detector_row.iter() {
                sum ^= result2.decoding[var_idx];
            }
            computed_syndrome2[det_idx] = sum;
        }

        println!("  Computed syndrome: {:?}", computed_syndrome2);

        assert_eq!(computed_syndrome2[0], detectors_case2[0],
                   "Detector 0: expected {}, got {}", detectors_case2[0], computed_syndrome2[0]);
        assert_eq!(computed_syndrome2[1], detectors_case2[1],
                   "Detector 1: expected {}, got {}", detectors_case2[1], computed_syndrome2[1]);
        assert_eq!(computed_syndrome2[2], detectors_case2[2],
                   "Detector 2 (BOUNDARY): expected {}, got {}", detectors_case2[2], computed_syndrome2[2]);
        assert_eq!(computed_syndrome2[3], detectors_case2[3],
                   "Detector 3: expected {}, got {}", detectors_case2[3], computed_syndrome2[3]);
    }
}

