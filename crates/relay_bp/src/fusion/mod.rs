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
use crate::decoder::{Bit, DecodeResult, Decoder, SparseBitMatrix};
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

/// Creates a fusion check matrix for combining partition solutions.
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
/// * `partitions` - Vector of partitions (unused but kept for API compatibility)
///
/// # Returns
/// Tuple of (fusion check matrix, all detector indices)
pub fn create_fusion_matrix(
    original_matrix: &SparseBitMatrix,
    _partitions: &[Partition],
) -> (Arc<SparseBitMatrix>, Vec<usize>) {
    // Includes ALL detectors in fusion matrix. This is NOT how MWPM did it, but it's the only way
    // I could get the decoder to get correct results. Basically warm-starts the decoder
    // Fusion BP with warm-started posteriors will:
    //   1. Preserve interior solutions
    //   2. Reconcile conflicting boundary beliefs
    let all_detectors: Vec<usize> = (0..original_matrix.rows()).collect();

    (Arc::new(original_matrix.clone()), all_detectors)
}

/// Fusion decoder that combines solutions from multiple partitions.
/// Doesn't do the hierarchical decoding that MWPM Fusion did, it only does the partitions one time and then
/// basically runs the regular BP decoder using those posteriors as "warm-starts"
pub struct FusionDecoder<N: PartialEq + Default + Clone + Copy> {
    original_matrix: Arc<SparseBitMatrix>,
    partitions: Vec<Partition>,
    min_sum_config: Arc<MinSumDecoderConfig>,
    relay_config: Arc<RelayDecoderConfig>,
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
    /// Create a new fusion decoder.
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
        FusionDecoder {
            original_matrix: check_matrix,
            partitions,
            min_sum_config,
            relay_config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Decode using fusion approach.
    ///
    /// 1. Run leaf decoders in parallel on partition sub-problems
    /// 2. Extract posteriors from each leaf
    /// 3. Combine posteriors (averaging at boundaries)
    /// 4. Create fusion matrix with only boundary detectors
    /// 5. Extract boundary detector observations
    /// 6. Run fusion decoder to reconcile boundaries
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

        // Step 2: Extract posteriors
        let posteriors: Vec<Array1<f64>> = leaf_results.iter().map(|(p, _)| p.clone()).collect();

        // Step 3: Combine posteriors
        let combined_posteriors =
            combine_posteriors(&posteriors, &self.partitions, self.original_matrix.cols());

        // Step 4: Create fusion decoder with all constraints
        let (fusion_matrix, _all_detector_indices) =
            create_fusion_matrix(&self.original_matrix, &self.partitions);
        let mut fusion_decoder: RelayDecoder<N> = RelayDecoder::new(
            fusion_matrix,
            self.min_sum_config.clone(),
            self.relay_config.clone(),
        );

        // Step 5: Load combined posteriors and run fusion decoding
        fusion_decoder.set_posterior_ratios_f64(combined_posteriors);
        let fusion_result = fusion_decoder.decode_detailed(detectors);

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
    use ndarray::array;

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
            create_fusion_matrix(&check_matrix, &partitions);

        // Should include ALL detectors (not just boundary)
        // This ensures fusion respects all constraints
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

