// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Comprehensive benchmark and validation suite for time-axis partitioning
//!
//! This benchmark validates the time-axis partitioning implementation by:
//! 1. Testing partition quality (boundary/volume ratios)
//! 2. Verifying correctness of decoding
//! 3. Comparing time-axis vs count-based partitioning
//! 4. Detecting potential reward-hacking in metrics

use ndarray::Array2;
use ndarray_npy::read_npy;
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::{RelayDecoder, RelayDecoderConfig};
use relay_bp::decoder::{Bit, Decoder, SparseBitMatrix};
use relay_bp::dem::DetectorErrorModel;
use relay_bp::fusion::{partition_by_count, partition_by_time_rounds, FusionDecoder, Partition};
use relay_bp::utilities::test::get_test_data_path;
use relay_bp::bipartite_graph::BipartiteGraph;
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
    println!("COMPREHENSIVE TIME-AXIS PARTITIONING BENCHMARK");
    println!("{}", "=".repeat(80));

    // Load [[72, 12, 6]] bivariate bicycle code with 6 rounds
    let resources = get_test_data_path();
    let code_path = resources.join("72_12_6_r6");

    println!("\n[1/6] Loading Test Data");
    println!("{}", "-".repeat(80));

    let code = DetectorErrorModel::load(code_path.clone())
        .expect("Failed to load quantum code");

    let test_detectors: Array2<Bit> =
        read_npy(resources.join("72_12_6_r6_detectors.npy"))
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
    // TEST 1: Partition Quality Analysis
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[2/6] PARTITION QUALITY ANALYSIS");
    println!("{}", "=".repeat(80));

    let mut partition_analyses = Vec::new();

    // Time-axis partitioning with different rounds_per_partition
    for &rounds_per_partition in &[1, 2, 3, 6] {
        let partitions = partition_by_time_rounds(
            &check_matrix,
            detectors_per_round,
            rounds_per_partition,
        );

        let analysis = analyze_partitions(
            &partitions,
            &check_matrix,
            format!("Time-axis ({} rounds/partition)", rounds_per_partition),
        );

        partition_analyses.push(analysis);
    }

    // Count-based partitioning with different partition counts
    for &num_partitions in &[2, 3, 6] {
        let partitions = partition_by_count(check_matrix.cols(), num_partitions);

        let analysis = analyze_partitions(
            &partitions,
            &check_matrix,
            format!("Count-based ({} partitions)", num_partitions),
        );

        partition_analyses.push(analysis);
    }

    // Print partition quality table
    println!("\nPartition Quality Results:");
    println!("{:-<100}", "");
    println!("{:<35} | {:>10} | {:>15} | {:>15} | {:>15}",
        "Strategy", "Partitions", "Avg Boundary/%", "Max Boundary/%", "Min Boundary/%");
    println!("{:-<100}", "");

    for analysis in &partition_analyses {
        println!("{:<35} | {:>10} | {:>14.1}% | {:>14.1}% | {:>14.1}%",
            analysis.strategy,
            analysis.num_partitions,
            analysis.avg_boundary_ratio * 100.0,
            analysis.max_boundary_ratio * 100.0,
            analysis.min_boundary_ratio * 100.0,
        );
    }
    println!("{:-<100}", "");

    // =============================================================================
    // TEST 2: Reward-Hacking Detection for Boundary Calculation
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[3/6] REWARD-HACKING VALIDATION");
    println!("{}", "=".repeat(80));

    println!("\nValidating boundary variable calculation...");

    // Test with a simple hand-crafted example
    let simple_validation = validate_boundary_calculation();
    if simple_validation {
        println!("  [PASS] Boundary calculation validated on hand-crafted example");
    } else {
        println!("  [FAIL] Boundary calculation has issues!");
        println!("  CRITICAL: This suggests reward-hacking in metrics");
    }

    // Check for suspicious patterns
    println!("\nChecking for reward-hacking patterns:");

    // Red flag 1: All strategies have similar ratios
    let time_axis_ratios: Vec<f64> = partition_analyses.iter()
        .filter(|a| a.strategy.starts_with("Time-axis"))
        .map(|a| a.avg_boundary_ratio)
        .collect();

    let count_based_ratios: Vec<f64> = partition_analyses.iter()
        .filter(|a| a.strategy.starts_with("Count-based"))
        .map(|a| a.avg_boundary_ratio)
        .collect();

    let time_avg = time_axis_ratios.iter().sum::<f64>() / time_axis_ratios.len() as f64;
    let count_avg = count_based_ratios.iter().sum::<f64>() / count_based_ratios.len() as f64;

    println!("  • Time-axis avg ratio: {:.3}", time_avg);
    println!("  • Count-based avg ratio: {:.3}", count_avg);

    if (time_avg - count_avg).abs() / count_avg < 0.1 {
        println!("  [WARNING] Time-axis and count-based have similar ratios!");
        println!("    This may indicate the metric is not measuring what we think");
    } else {
        println!("  [PASS] Clear difference between strategies");
    }

    // Red flag 2: Time-axis ratio is too high
    if time_avg > 0.5 {
        println!("  [WARNING] Time-axis boundary ratio > 0.5 - this seems wrong!");
    } else if time_avg < 0.15 {
        println!("  [PASS] Time-axis boundary ratio < 0.15 (good partitioning)");
    } else {
        println!("  [OK] Time-axis boundary ratio is reasonable");
    }

    // =============================================================================
    // TEST 3: Correctness Validation
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[4/6] CORRECTNESS VALIDATION");
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

    // Time-axis fusion decoder (2 rounds/partition)
    println!("  Running time-axis fusion decoder (2 rounds/partition)...");
    let time_axis_partitions = partition_by_time_rounds(&check_matrix, detectors_per_round, 2);
    let time_axis_metrics = run_fusion_decoder(
        &check_matrix,
        &test_detectors,
        num_test_samples,
        &bp_config,
        &relay_config,
        time_axis_partitions,
    );

    // Count-based fusion decoder (3 partitions)
    println!("  Running count-based fusion decoder (3 partitions)...");
    let count_based_partitions = partition_by_count(check_matrix.cols(), 3);
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

    // Check for identical decodings (potential reward-hacking)
    let identical_time_axis = count_identical_decodings(&baseline_metrics.decodings, &time_axis_metrics.decodings);
    let identical_count_based = count_identical_decodings(&baseline_metrics.decodings, &count_based_metrics.decodings);

    println!("\nDecoding Agreement:");
    println!("  • Time-axis vs Baseline: {}/{} identical", identical_time_axis, num_test_samples);
    println!("  • Count-based vs Baseline: {}/{} identical", identical_count_based, num_test_samples);

    // =============================================================================
    // TEST 4: Detailed Partition Structure Validation
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[5/6] PARTITION STRUCTURE VALIDATION");
    println!("{}", "=".repeat(80));

    println!("\nValidating time-axis partitioning with 2 rounds/partition:");

    let partitions_2round = partition_by_time_rounds(&check_matrix, detectors_per_round, 2);

    println!("  Expected partitions: {}", total_rounds / 2);
    println!("  Actual partitions: {}", partitions_2round.len());

    if partitions_2round.len() == total_rounds / 2 {
        println!("  [PASS] Correct number of partitions");
    } else {
        println!("  [FAIL] Unexpected number of partitions!");
    }

    // Verify partition coverage
    let mut all_detectors_covered = HashSet::new();
    let mut all_variables_seen = HashSet::new();

    for (i, partition) in partitions_2round.iter().enumerate() {
        println!("\n  Partition {}:", i);
        println!("    Detectors: {} (expected: {})",
            partition.detector_indices.len(),
            2 * detectors_per_round
        );
        println!("    Variables: {}", partition.variable_indices.len());

        // Check detector range
        if !partition.detector_indices.is_empty() {
            let min_det = *partition.detector_indices.iter().min().unwrap();
            let max_det = *partition.detector_indices.iter().max().unwrap();
            let expected_min = i * 2 * detectors_per_round;
            let expected_max = expected_min + 2 * detectors_per_round - 1;

            println!("    Detector range: [{}..{}] (expected: [{}..{}])",
                min_det, max_det, expected_min, expected_max);

            if min_det == expected_min && max_det == expected_max {
                println!("    [PASS] Correct detector range");
            } else {
                println!("    [FAIL] Unexpected detector range!");
            }
        }

        // Track coverage
        for &det in &partition.detector_indices {
            if !all_detectors_covered.insert(det) {
                println!("    [FAIL] Detector {} appears in multiple partitions!", det);
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
    // TEST 5: Final Report and Recommendations
    // =============================================================================

    println!("\n{}", "=".repeat(80));
    println!("[6/6] SUMMARY AND RECOMMENDATIONS");
    println!("{}", "=".repeat(80));

    println!("\n=== KEY FINDINGS ===\n");

    // Finding 1: Partition quality improvement
    let best_time_axis = partition_analyses.iter()
        .filter(|a| a.strategy.starts_with("Time-axis"))
        .min_by(|a, b| a.avg_boundary_ratio.partial_cmp(&b.avg_boundary_ratio).unwrap())
        .unwrap();

    let best_count_based = partition_analyses.iter()
        .filter(|a| a.strategy.starts_with("Count-based"))
        .min_by(|a, b| a.avg_boundary_ratio.partial_cmp(&b.avg_boundary_ratio).unwrap())
        .unwrap();

    let improvement_factor = best_count_based.avg_boundary_ratio / best_time_axis.avg_boundary_ratio;

    println!("1. PARTITION QUALITY:");
    println!("   • Best time-axis: {:.1}% boundary ratio ({})",
        best_time_axis.avg_boundary_ratio * 100.0,
        best_time_axis.strategy);
    println!("   • Best count-based: {:.1}% boundary ratio ({})",
        best_count_based.avg_boundary_ratio * 100.0,
        best_count_based.strategy);
    println!("   • Improvement factor: {:.1}x better with time-axis partitioning", improvement_factor);

    // Finding 2: Correctness
    println!("\n2. CORRECTNESS:");
    if baseline_metrics.success_count == time_axis_metrics.success_count {
        println!("   • [PASS] Time-axis fusion matches baseline success rate");
    } else {
        println!("   • [WARNING] Time-axis fusion has different success rate than baseline");
        println!("     Baseline: {}/{}", baseline_metrics.success_count, num_test_samples);
        println!("     Time-axis: {}/{}", time_axis_metrics.success_count, num_test_samples);
    }

    if time_axis_metrics.syndrome_satisfaction_count == num_test_samples {
        println!("   • [PASS] All time-axis decodings satisfy syndromes");
    } else {
        println!("   • [FAIL] Some time-axis decodings do NOT satisfy syndromes!");
        println!("     This is a critical correctness issue!");
    }

    // Finding 3: Partitioning effectiveness
    println!("\n3. TIME-AXIS PARTITIONING EFFECTIVENESS:");
    if best_time_axis.avg_boundary_ratio < 0.15 {
        println!("   • [EXCELLENT] Boundary ratio < 15% demonstrates effective partitioning");
    } else if best_time_axis.avg_boundary_ratio < 0.25 {
        println!("   • [GOOD] Boundary ratio < 25% shows reasonable partitioning");
    } else {
        println!("   • [POOR] Boundary ratio >= 25% suggests partitioning may not help");
    }

    println!("\n=== VALIDATION STATUS ===\n");

    let mut validation_passed = true;

    // Check 1: Boundary calculation
    if simple_validation {
        println!("[PASS] Boundary calculation is correct");
    } else {
        println!("[FAIL] Boundary calculation has issues");
        validation_passed = false;
    }

    // Check 2: Partition coverage
    if all_detectors_covered.len() == total_detectors {
        println!("[PASS] Partition coverage is complete");
    } else {
        println!("[FAIL] Partition coverage is incomplete");
        validation_passed = false;
    }

    // Check 3: Syndrome satisfaction
    if time_axis_metrics.syndrome_satisfaction_count == num_test_samples {
        println!("[PASS] All fusion decodings satisfy syndromes");
    } else {
        println!("[FAIL] Some fusion decodings do not satisfy syndromes");
        validation_passed = false;
    }

    // Check 4: Improvement vs count-based
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
        println!("TIME-AXIS PARTITIONING: VALIDATED");
        println!("\nThe implementation correctly partitions by time and achieves");
        println!("better boundary/volume ratios than naive count-based partitioning.");
        println!("\nRECOMMENDATION: Proceed with using time-axis partitioning for");
        println!("spacetime quantum error correction codes.");
    } else {
        println!("TIME-AXIS PARTITIONING: VALIDATION FAILED");
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

/// Count boundary variables in a partition
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

    let mut boundary_vars = HashSet::new();

    // For each variable in this partition, check if it connects to external detectors
    for &var_idx in &this_partition.variable_indices {
        // Find which detectors use this variable
        for det_idx in 0..check_matrix.rows() {
            let row = check_matrix.outer_view(det_idx).unwrap();
            let connects_to_var = row.iter().any(|(col, _)| col == var_idx);

            if connects_to_var && !det_set.contains(&det_idx) {
                // This variable connects to a detector outside the partition
                boundary_vars.insert(var_idx);
                break;
            }
        }
    }

    boundary_vars.len()
}

/// Validate boundary calculation with a simple hand-crafted example
fn validate_boundary_calculation() -> bool {
    use ndarray::array;

    // Create a simple chain code:
    // det0 -- var0,var1 -- det1 -- var2,var3 -- det2 -- var4,var5 -- det3
    let dense = array![
        [1, 1, 0, 0, 0, 0], // det0: vars 0,1
        [0, 1, 1, 0, 0, 0], // det1: vars 1,2 (boundary!)
        [0, 0, 1, 1, 0, 0], // det2: vars 2,3
        [0, 0, 0, 1, 1, 0], // det3: vars 3,4 (boundary!)
        [0, 0, 0, 0, 1, 1], // det4: vars 4,5
    ];

    let check_matrix = SparseBitMatrix::from_dense(dense);

    // Partition: [vars 0,1,2] and [vars 3,4,5]
    // Detectors for partition 0: 0,1,2 (connect to vars 0,1,2)
    // Detectors for partition 1: 3,4 (connect to vars 3,4,5)
    // Boundary vars in partition 0: var2 (connects to det2 which spans both partitions)
    // Boundary vars in partition 1: var3 (connects to det2 which spans both partitions)

    let partitions = vec![
        Partition {
            variable_indices: vec![0, 1, 2],
            detector_indices: vec![0, 1, 2],
        },
        Partition {
            variable_indices: vec![3, 4, 5],
            detector_indices: vec![2, 3, 4],
        },
    ];

    let boundary_0 = count_boundary_variables(&partitions, 0, &check_matrix);
    let boundary_1 = count_boundary_variables(&partitions, 1, &check_matrix);

    // Expect var2 to be boundary for partition 0 (connects to det2 which is in both)
    // Actually, detector 2 is IN partition 0's detector list, so var2 should NOT be boundary
    // Let me reconsider...

    // Actually, with the detector lists as defined:
    // - Partition 0 has detectors [0,1,2], variables [0,1,2]
    // - Partition 1 has detectors [2,3,4], variables [3,4,5]

    // For partition 0:
    // - var0: connects to det0 (in partition) -> NOT boundary
    // - var1: connects to det0,1 (both in partition) -> NOT boundary
    // - var2: connects to det1,2 (both in partition) -> NOT boundary

    // For partition 1:
    // - var3: connects to det2,3 (both in partition) -> NOT boundary
    // - var4: connects to det3,4 (both in partition) -> NOT boundary
    // - var5: connects to det4 (in partition) -> NOT boundary

    // Hmm, with overlapping detector sets, there are NO boundary variables!
    // Let me fix the test case with non-overlapping detector sets

    let partitions_fixed = vec![
        Partition {
            variable_indices: vec![0, 1, 2],
            detector_indices: vec![0, 1],
        },
        Partition {
            variable_indices: vec![3, 4, 5],
            detector_indices: vec![3, 4],
        },
    ];

    let boundary_0_fixed = count_boundary_variables(&partitions_fixed, 0, &check_matrix);
    let boundary_1_fixed = count_boundary_variables(&partitions_fixed, 1, &check_matrix);

    // Now:
    // Partition 0: detectors [0,1], variables [0,1,2]
    // - var2: connects to det1,2; det2 is NOT in partition -> BOUNDARY
    // Expected: 1 boundary variable

    // Partition 1: detectors [3,4], variables [3,4,5]
    // - var3: connects to det2,3; det2 is NOT in partition -> BOUNDARY
    // Expected: 1 boundary variable

    let validation_passed = boundary_0_fixed == 1 && boundary_1_fixed == 1;

    if !validation_passed {
        println!("  Expected partition 0 boundary: 1, got: {}", boundary_0_fixed);
        println!("  Expected partition 1 boundary: 1, got: {}", boundary_1_fixed);
    }

    validation_passed
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
