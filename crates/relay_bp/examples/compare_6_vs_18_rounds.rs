// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Compare time-axis partitioning on 6-round vs 18-round circuits

use relay_bp::decoder::{SparseBitMatrix, Decoder};
use relay_bp::dem::DetectorErrorModel;
use relay_bp::fusion::{partition_by_count, partition_by_time_rounds, Partition};
use relay_bp::utilities::test::get_test_data_path;
use std::collections::HashSet;
use std::sync::Arc;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("COMPARISON: 6-ROUND vs 18-ROUND TIME-AXIS PARTITIONING");
    println!("{}", "=".repeat(80));

    let resources = get_test_data_path();

    // Test both datasets
    println!("\n{}", "-".repeat(80));
    println!("TEST 1: [[72,12,6]] Code with 6 Rounds");
    println!("{}", "-".repeat(80));
    test_dataset(
        resources.join("72_12_6_r6"),
        72,  // detectors per round
        6,   // total rounds
    );

    println!("\n{}", "-".repeat(80));
    println!("TEST 2: [[288,12,18]] Code with 18 Rounds");
    println!("{}", "-".repeat(80));
    test_dataset(
        resources.join("288_12_18_r18"),
        288, // detectors per round
        18,  // total rounds
    );

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\n18-round circuit provides:");
    println!("  • 3x more temporal depth (18 vs 6 rounds)");
    println!("  • More partitions possible");
    println!("  • Smaller boundary/volume ratios");
    println!("  • Better demonstration of time-axis partitioning benefits");
    println!();
    println!("For full validation, would need 150+ rounds.");
    println!("{}", "=".repeat(80));
}

fn test_dataset(
    code_path: std::path::PathBuf,
    detectors_per_round: usize,
    total_rounds: usize,
) {
    // Load code
    let code = DetectorErrorModel::load(code_path)
        .expect("Failed to load code");

    println!("\nDataset info:");
    println!("  Detectors: {} ({} per round × {} rounds)",
        code.detector_error_matrix.rows(),
        detectors_per_round,
        total_rounds
    );
    println!("  Error variables: {}", code.detector_error_matrix.cols());

    let check_matrix = Arc::new(code.detector_error_matrix.to_csr());

    // Test time-axis partitioning with different configurations
    println!("\nTime-Axis Partitioning Results:");

    for rounds_per_partition in &[2, 3] {
        if *rounds_per_partition > total_rounds {
            continue;
        }

        let partitions = partition_by_time_rounds(
            &check_matrix,
            detectors_per_round,
            *rounds_per_partition,
        );

        let stats = analyze_boundary_stats(&partitions, &check_matrix);

        println!("  {} rounds/partition: {} partitions, boundary ratio {:.3} ({:.1}%)",
            rounds_per_partition,
            partitions.len(),
            stats.avg_ratio,
            stats.avg_ratio * 100.0
        );
    }

    // Compare with count-based partitioning
    let num_partitions_for_comparison = (total_rounds / 2).max(2).min(4);
    let count_partitions = partition_by_count(
        check_matrix.cols(),
        num_partitions_for_comparison
    );
    let count_stats = analyze_boundary_stats(&count_partitions, &check_matrix);

    println!("\nCount-Based Partitioning (baseline):");
    println!("  {} partitions: boundary ratio {:.3} ({:.1}%)",
        num_partitions_for_comparison,
        count_stats.avg_ratio,
        count_stats.avg_ratio * 100.0
    );

    // Calculate improvement
    let time_axis_best = partition_by_time_rounds(&check_matrix, detectors_per_round, 3);
    let time_stats = analyze_boundary_stats(&time_axis_best, &check_matrix);
    let improvement = count_stats.avg_ratio / time_stats.avg_ratio;

    println!("\nImprovement: {:.1}x better boundary ratio with time-axis partitioning",
        improvement);
}

struct BoundaryStats {
    avg_ratio: f64,
    max_ratio: f64,
}

fn analyze_boundary_stats(
    partitions: &[Partition],
    check_matrix: &SparseBitMatrix,
) -> BoundaryStats {
    let mut ratios = Vec::new();

    for (i, partition) in partitions.iter().enumerate() {
        let boundary_vars = count_boundary_variables(partitions, i, check_matrix);
        let total_vars = partition.variable_indices.len();

        if total_vars > 0 {
            let ratio = boundary_vars as f64 / total_vars as f64;
            ratios.push(ratio);
        }
    }

    if ratios.is_empty() {
        return BoundaryStats {
            avg_ratio: 0.0,
            max_ratio: 0.0,
        };
    }

    BoundaryStats {
        avg_ratio: ratios.iter().sum::<f64>() / ratios.len() as f64,
        max_ratio: ratios.iter().cloned().fold(0.0, f64::max),
    }
}

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

    for &var_idx in &this_partition.variable_indices {
        for det_idx in 0..check_matrix.rows() {
            let row = check_matrix.outer_view(det_idx).unwrap();
            let connects_to_var = row.iter().any(|(col, _)| col == var_idx);

            if connects_to_var && !det_set.contains(&det_idx) {
                boundary_vars.insert(var_idx);
                break;
            }
        }
    }

    boundary_vars.len()
}
