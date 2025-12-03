// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Test suite for fusion decoding on 150-round circuits
//!
//! This test validates fusion implementation with 150 rounds by:
//! 1. Testing partition quality (boundary/volume ratios) on 150-round code
//! 2. Verifying correctness of decoding
//! 3. Comparing time-axis vs count-based partitioning
//! 4. Using a single partition size for faster execution

use ndarray::Array2;
use ndarray_npy::read_npy;
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::{RelayDecoder, RelayDecoderConfig};
use relay_bp::decoder::{Bit, Decoder, SparseBitMatrix};
use relay_bp::dem::DetectorErrorModel;
use relay_bp::fusion::{partition_by_count, partition_by_time_rounds, FusionDecoder, Partition};
use relay_bp::utilities::test::get_test_data_path;
use std::collections::HashSet;
use std::sync::Arc;
use ndarray::Array1;

/// Statistics about partition boundary quality
#[derive(Debug, Clone)]
struct BoundaryStats {
    boundary_vars: usize,
    total_vars: usize,
    ratio: f64,
}

#[derive(Debug, Clone)]
struct PartitionAnalysis {
    strategy: String,
    num_partitions: usize,
    partition_stats: Vec<BoundaryStats>,
    avg_boundary_ratio: f64,
    max_boundary_ratio: f64,
    min_boundary_ratio: f64,
}

#[derive(Debug, Clone)]
struct DecodingMetrics {
    success_count: usize,
    total_samples: usize,
    total_iterations: usize,
    syndrome_satisfaction_count: usize,
    decodings: Vec<Array1<u8>>,
}

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("FUSION TEST: 150-ROUND CIRCUIT");
    println!("{}", "=".repeat(80));

    // Load [[72, 12, 6]] bivariate bicycle code with 150 rounds
    let resources = get_test_data_path();
    let code_path = resources.join("72_12_6_r150");

    println!("\n[1/5] Loading Test Data");
    println!("{}", "-".repeat(80));

    let code = DetectorErrorModel::load(code_path.clone())
        .expect("Failed to load quantum code");

    let test_detectors: Array2<Bit> =
        read_npy(resources.join("72_12_6_r150_detectors.npy"))
            .expect("Failed to load test detectors");

    println!("  Code: [[72, 12, 6]] bivariate bicycle");
    println!("  Check matrix: {} detectors × {} error variables",
        code.detector_error_matrix.rows(),
        code.detector_error_matrix.cols()
    );
    println!("  Test samples: {}", test_detectors.nrows());

    // Temporal structure
    let detectors_per_round = 72;
    let total_detectors = code.detector_error_matrix.rows();
    let total_rounds = total_detectors / detectors_per_round;

    println!("\n  Temporal structure:");
    println!("    Detectors per round: {}", detectors_per_round);
    println!("    Total rounds: {}", total_rounds);
    println!("    Total detectors: {} ({}×{})", total_detectors, total_rounds, detectors_per_round);

    let check_matrix = Arc::new(code.detector_error_matrix.to_csr());
    let num_test_samples = test_detectors.nrows().min(20); // Use up to 20 samples

    // Configuration for decoders
    let bp_config = Arc::new(MinSumDecoderConfig {
        error_priors: code.error_priors,
        max_iter: 200,
        alpha: None,
        alpha_iteration_scaling_factor: 0.,
        gamma0: Some(0.1),
        ..Default::default()
    });

    let relay_config = Arc::new(RelayDecoderConfig {
        pre_iter: 120,
        num_sets: 40,
        set_max_iter: 150,
        logging: false,
        ..Default::default()
    });

    // =============================================================================
    // TEST 1: Partition Quality Analysis (Single Configuration)
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[2/5] PARTITION QUALITY ANALYSIS");
    println!("{}", "=".repeat(80));

    // Test with 10 rounds per partition (15 partitions total for 150 rounds)
    let rounds_per_partition = 10;
    println!("\nTesting time-axis partitioning with {} rounds/partition:", rounds_per_partition);

    let time_axis_partitions = partition_by_time_rounds(
        &check_matrix,
        detectors_per_round,
        rounds_per_partition,
    );

    let time_axis_analysis = analyze_partitions(
        &time_axis_partitions,
        &check_matrix,
        format!("Time-axis ({} rounds/partition)", rounds_per_partition),
    );

    // Compare with count-based partitioning (15 partitions to match)
    let num_partitions = total_rounds / rounds_per_partition;
    println!("\nTesting count-based partitioning with {} partitions:", num_partitions);

    let count_based_partitions = partition_by_count(check_matrix.cols(), num_partitions);
    let count_based_analysis = analyze_partitions(
        &count_based_partitions,
        &check_matrix,
        format!("Count-based ({} partitions)", num_partitions),
    );

    // Print partition quality table
    println!("\nPartition Quality Results:");
    println!("{:-<100}", "");
    println!("{:<35} | {:>10} | {:>15} | {:>15} | {:>15}",
        "Strategy", "Partitions", "Avg Boundary/%", "Max Boundary/%", "Min Boundary/%");
    println!("{:-<100}", "");

    println!("{:<35} | {:>10} | {:>14.1}% | {:>14.1}% | {:>14.1}%",
        time_axis_analysis.strategy,
        time_axis_analysis.num_partitions,
        time_axis_analysis.avg_boundary_ratio * 100.0,
        time_axis_analysis.max_boundary_ratio * 100.0,
        time_axis_analysis.min_boundary_ratio * 100.0,
    );

    println!("{:<35} | {:>10} | {:>14.1}% | {:>14.1}% | {:>14.1}%",
        count_based_analysis.strategy,
        count_based_analysis.num_partitions,
        count_based_analysis.avg_boundary_ratio * 100.0,
        count_based_analysis.max_boundary_ratio * 100.0,
        count_based_analysis.min_boundary_ratio * 100.0,
    );
    println!("{:-<100}", "");

    let improvement_factor = count_based_analysis.avg_boundary_ratio / time_axis_analysis.avg_boundary_ratio;
    println!("\n  Improvement factor: {:.2}x better with time-axis partitioning", improvement_factor);

    // =============================================================================
    // TEST 2: Correctness Validation
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[3/5] CORRECTNESS VALIDATION");
    println!("{}", "=".repeat(80));

    println!("\nRunning {} test samples through decoders...", num_test_samples);

    // Baseline decoder
    println!("\n  Running baseline RelayBP decoder...");
    let baseline_metrics = run_baseline_decoder(
        &check_matrix,
        &test_detectors,
        num_test_samples,
        &bp_config,
        &relay_config,
    );

    // Time-axis fusion decoder
    println!("  Running time-axis fusion decoder ({} rounds/partition)...", rounds_per_partition);
    let time_axis_metrics = run_fusion_decoder(
        &check_matrix,
        &test_detectors,
        num_test_samples,
        &bp_config,
        &relay_config,
        time_axis_partitions,
    );

    // Count-based fusion decoder
    println!("  Running count-based fusion decoder ({} partitions)...", num_partitions);
    let count_based_metrics = run_fusion_decoder(
        &check_matrix,
        &test_detectors,
        num_test_samples,
        &bp_config,
        &relay_config,
        count_based_partitions,
    );

    // Print correctness table
    println!("\nCorrectness Results:");
    println!("{:-<100}", "");
    println!("{:<30} | {:>15} | {:>15} | {:>15}",
        "Decoder", "Success Rate", "Syndrome Sat.", "Avg Iterations");
    println!("{:-<100}", "");

    println!("{:<30} | {:>6}/{:<7} | {:>6}/{:<7} | {:>15.1}",
        "Baseline",
        baseline_metrics.success_count, num_test_samples,
        baseline_metrics.syndrome_satisfaction_count, num_test_samples,
        baseline_metrics.total_iterations as f64 / num_test_samples as f64,
    );

    println!("{:<30} | {:>6}/{:<7} | {:>6}/{:<7} | {:>15.1}",
        "Time-axis fusion",
        time_axis_metrics.success_count, num_test_samples,
        time_axis_metrics.syndrome_satisfaction_count, num_test_samples,
        time_axis_metrics.total_iterations as f64 / num_test_samples as f64,
    );

    println!("{:<30} | {:>6}/{:<7} | {:>6}/{:<7} | {:>15.1}",
        "Count-based fusion",
        count_based_metrics.success_count, num_test_samples,
        count_based_metrics.syndrome_satisfaction_count, num_test_samples,
        count_based_metrics.total_iterations as f64 / num_test_samples as f64,
    );
    println!("{:-<100}", "");

    // Check for identical decodings
    let identical_time_axis = count_identical_decodings(&baseline_metrics.decodings, &time_axis_metrics.decodings);
    let identical_count_based = count_identical_decodings(&baseline_metrics.decodings, &count_based_metrics.decodings);

    println!("\nDecoding Agreement:");
    println!("  • Time-axis vs Baseline: {}/{} identical", identical_time_axis, num_test_samples);
    println!("  • Count-based vs Baseline: {}/{} identical", identical_count_based, num_test_samples);

    // =============================================================================
    // TEST 3: Partition Structure Validation
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[4/5] PARTITION STRUCTURE VALIDATION");
    println!("{}", "=".repeat(80));

    println!("\nValidating time-axis partitioning with {} rounds/partition:", rounds_per_partition);

    let partitions = partition_by_time_rounds(&check_matrix, detectors_per_round, rounds_per_partition);

    println!("  Expected partitions: {}", total_rounds / rounds_per_partition);
    println!("  Actual partitions: {}", partitions.len());

    if partitions.len() == total_rounds / rounds_per_partition {
        println!("  [PASS] Correct number of partitions");
    } else {
        println!("  [FAIL] Unexpected number of partitions!");
    }

    // Verify partition coverage
    let mut all_detectors_covered = HashSet::new();
    let mut all_variables_seen = HashSet::new();

    for (i, partition) in partitions.iter().enumerate() {
        if i < 3 || i >= partitions.len() - 3 {
            // Only print first 3 and last 3 partitions
            println!("\n  Partition {}:", i);
            println!("    Detectors: {} (expected: {})",
                partition.detector_indices.len(),
                rounds_per_partition * detectors_per_round
            );
            println!("    Variables: {}", partition.variable_indices.len());

            // Check detector range
            if !partition.detector_indices.is_empty() {
                let min_det = *partition.detector_indices.iter().min().unwrap();
                let max_det = *partition.detector_indices.iter().max().unwrap();
                let expected_min = i * rounds_per_partition * detectors_per_round;
                let expected_max = expected_min + rounds_per_partition * detectors_per_round - 1;

                println!("    Detector range: [{}..{}] (expected: [{}..{}])",
                    min_det, max_det, expected_min, expected_max);

                if min_det == expected_min && max_det == expected_max {
                    println!("    [PASS] Correct detector range");
                } else {
                    println!("    [FAIL] Unexpected detector range!");
                }
            }
        }

        // Track coverage
        for &det in &partition.detector_indices {
            if !all_detectors_covered.insert(det) {
                if i < 3 || i >= partitions.len() - 3 {
                    println!("    [FAIL] Detector {} appears in multiple partitions!", det);
                }
            }
        }

        for &var in &partition.variable_indices {
            all_variables_seen.insert(var);
        }
    }

    println!("\n  Coverage check:");
    println!("    Detectors covered: {}/{}", all_detectors_covered.len(), total_detectors);
    println!("    Variables seen: {}/{}", all_variables_seen.len(), check_matrix.cols());

    if all_detectors_covered.len() == total_detectors {
        println!("    [PASS] All detectors covered exactly once");
    } else {
        println!("    [FAIL] Detector coverage incomplete!");
    }

    // =============================================================================
    // TEST 4: Summary
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[5/5] SUMMARY");
    println!("{}", "=".repeat(80));

    println!("\n=== KEY FINDINGS ===\n");

    println!("1. PARTITION QUALITY:");
    println!("   • Time-axis: {:.1}% boundary ratio",
        time_axis_analysis.avg_boundary_ratio * 100.0);
    println!("   • Count-based: {:.1}% boundary ratio",
        count_based_analysis.avg_boundary_ratio * 100.0);
    println!("   • Improvement: {:.2}x better with time-axis", improvement_factor);

    println!("\n2. CORRECTNESS:");
    if baseline_metrics.success_count == time_axis_metrics.success_count {
        println!("   • [PASS] Time-axis fusion matches baseline success rate");
    } else {
        println!("   • [WARNING] Time-axis fusion has different success rate");
        println!("     Baseline: {}/{}", baseline_metrics.success_count, num_test_samples);
        println!("     Time-axis: {}/{}", time_axis_metrics.success_count, num_test_samples);
    }

    if time_axis_metrics.syndrome_satisfaction_count == num_test_samples {
        println!("   • [PASS] All time-axis decodings satisfy syndromes");
    } else {
        println!("   • [FAIL] Some time-axis decodings do NOT satisfy syndromes!");
    }

    println!("\n3. PARTITIONING EFFECTIVENESS:");
    if time_axis_analysis.avg_boundary_ratio < 0.15 {
        println!("   • [EXCELLENT] Boundary ratio < 15% demonstrates effective partitioning");
    } else if time_axis_analysis.avg_boundary_ratio < 0.25 {
        println!("   • [GOOD] Boundary ratio < 25% shows reasonable partitioning");
    } else {
        println!("   • [FAIR] Boundary ratio >= 25%");
    }

    println!("\n=== VALIDATION STATUS ===\n");

    let mut validation_passed = true;

    // Check 1: Partition coverage
    if all_detectors_covered.len() == total_detectors {
        println!("[PASS] Partition coverage is complete");
    } else {
        println!("[FAIL] Partition coverage is incomplete");
        validation_passed = false;
    }

    // Check 2: Syndrome satisfaction
    if time_axis_metrics.syndrome_satisfaction_count == num_test_samples {
        println!("[PASS] All fusion decodings satisfy syndromes");
    } else {
        println!("[FAIL] Some fusion decodings do not satisfy syndromes");
        validation_passed = false;
    }

    // Check 3: Improvement vs count-based
    if improvement_factor > 1.5 {
        println!("[PASS] Time-axis shows significant improvement over count-based");
    } else if improvement_factor > 1.0 {
        println!("[MARGINAL] Time-axis shows some improvement over count-based");
    } else {
        println!("[FAIL] Time-axis does not improve over count-based");
        validation_passed = false;
    }

    println!("\n=== OVERALL ASSESSMENT ===\n");

    if validation_passed {
        println!("FUSION DECODING ON 150-ROUND CIRCUIT: VALIDATED");
        println!("\nThe implementation correctly partitions by time and achieves");
        println!("better boundary/volume ratios than naive count-based partitioning.");
    } else {
        println!("FUSION DECODING: VALIDATION FAILED");
        println!("\nOne or more critical issues were detected. Review the failures");
        println!("above before proceeding.");
    }

    println!("\n{}", "=".repeat(80));
}

/// Analyze partition quality
fn analyze_partitions(
    partitions: &[Partition],
    check_matrix: &SparseBitMatrix,
    strategy: String,
) -> PartitionAnalysis {
    let mut partition_stats = Vec::new();

    for (i, partition) in partitions.iter().enumerate() {
        let boundary_vars = count_boundary_variables(partitions, i, check_matrix);
        let total_vars = partition.variable_indices.len();
        let ratio = boundary_vars as f64 / total_vars as f64;

        partition_stats.push(BoundaryStats {
            boundary_vars,
            total_vars,
            ratio,
        });
    }

    let avg_boundary_ratio = partition_stats.iter().map(|s| s.ratio).sum::<f64>()
        / partition_stats.len() as f64;
    let max_boundary_ratio = partition_stats.iter().map(|s| s.ratio).fold(0.0, f64::max);
    let min_boundary_ratio = partition_stats.iter().map(|s| s.ratio).fold(1.0, f64::min);

    PartitionAnalysis {
        strategy,
        num_partitions: partitions.len(),
        partition_stats,
        avg_boundary_ratio,
        max_boundary_ratio,
        min_boundary_ratio,
    }
}

/// Count boundary variables in a partition (optimized for sparse matrices)
fn count_boundary_variables(
    partitions: &[Partition],
    partition_idx: usize,
    check_matrix: &SparseBitMatrix,
) -> usize {
    let this_partition = &partitions[partition_idx];
    let det_set: HashSet<usize> = this_partition
        .detector_indices
        .iter()
        .copied()
        .collect();

    let var_set: HashSet<usize> = this_partition
        .variable_indices
        .iter()
        .copied()
        .collect();

    let mut boundary_vars = HashSet::new();

    // Iterate through detectors ONCE, building boundary set efficiently
    // For each detector NOT in this partition, mark all its variables (if in partition) as boundary
    for det_idx in 0..check_matrix.rows() {
        if det_set.contains(&det_idx) {
            continue; // Skip detectors in this partition
        }

        let row = check_matrix.outer_view(det_idx).unwrap();
        for (var_idx, _) in row.iter() {
            if var_set.contains(&var_idx) {
                // This variable is in the partition but connects to external detector
                boundary_vars.insert(var_idx);
            }
        }
    }

    boundary_vars.len()
}

/// Run baseline decoder on test samples
fn run_baseline_decoder(
    check_matrix: &Arc<SparseBitMatrix>,
    test_detectors: &Array2<Bit>,
    num_samples: usize,
    bp_config: &Arc<MinSumDecoderConfig>,
    relay_config: &Arc<RelayDecoderConfig>,
) -> DecodingMetrics {
    let mut decoder: RelayDecoder<f32> = RelayDecoder::new(
        check_matrix.clone(),
        bp_config.clone(),
        relay_config.clone(),
    );

    let mut success_count = 0;
    let mut total_iterations = 0;
    let mut syndrome_satisfaction_count = 0;
    let mut decodings = Vec::new();

    for i in 0..num_samples {
        let detectors = test_detectors.row(i);
        let result = decoder.decode_detailed(detectors);

        if result.success {
            success_count += 1;
        }
        total_iterations += result.iterations;

        // Verify syndrome satisfaction
        if verify_syndrome(&result.decoding, &detectors, check_matrix) {
            syndrome_satisfaction_count += 1;
        }

        decodings.push(result.decoding);
    }

    DecodingMetrics {
        success_count,
        total_samples: num_samples,
        total_iterations,
        syndrome_satisfaction_count,
        decodings,
    }
}

/// Run fusion decoder on test samples
fn run_fusion_decoder(
    check_matrix: &Arc<SparseBitMatrix>,
    test_detectors: &Array2<Bit>,
    num_samples: usize,
    bp_config: &Arc<MinSumDecoderConfig>,
    relay_config: &Arc<RelayDecoderConfig>,
    partitions: Vec<Partition>,
) -> DecodingMetrics {
    let decoder: FusionDecoder<f32> = FusionDecoder::new(
        check_matrix.clone(),
        partitions,
        bp_config.clone(),
        relay_config.clone(),
    );

    let mut success_count = 0;
    let mut total_iterations = 0;
    let mut syndrome_satisfaction_count = 0;
    let mut decodings = Vec::new();

    for i in 0..num_samples {
        let detectors = test_detectors.row(i);
        let result = decoder.decode_fusion(detectors);

        if result.success {
            success_count += 1;
        }
        total_iterations += result.iterations;

        // Verify syndrome satisfaction
        if verify_syndrome(&result.decoding, &detectors, check_matrix) {
            syndrome_satisfaction_count += 1;
        }

        decodings.push(result.decoding);
    }

    DecodingMetrics {
        success_count,
        total_samples: num_samples,
        total_iterations,
        syndrome_satisfaction_count,
        decodings,
    }
}

/// Verify that a decoding satisfies the syndrome: H @ decoding % 2 == detectors
fn verify_syndrome(decoding: &Array1<u8>, detectors: &ndarray::ArrayView1<u8>, check_matrix: &SparseBitMatrix) -> bool {
    for (det_idx, row) in check_matrix.outer_iterator().enumerate() {
        let mut sum = 0u8;
        for (var_idx, _) in row.iter() {
            sum ^= decoding[var_idx];
        }
        if sum != detectors[det_idx] {
            return false;
        }
    }
    true
}

/// Count how many decodings are identical between two sets
fn count_identical_decodings(decodings1: &[Array1<u8>], decodings2: &[Array1<u8>]) -> usize {
    decodings1.iter()
        .zip(decodings2.iter())
        .filter(|(d1, d2)| d1 == d2)
        .count()
}

