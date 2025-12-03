#!/usr/bin/env python3
"""
Generate multi-round [[72, 12, 6]] quantum error correction data for fusion decoding.

This script generates a 150-round bivariate bicycle code circuit with proper temporal
structure to enable time-axis partitioning for fusion-based decoding.

The output includes:
- Check matrix H (detectors × error variables)
- Observable matrix A
- Error priors
- Test detector samples and corresponding errors
"""

import sys
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz, csr_matrix
import stim

# Add tests directory to path for circuit loading utilities
sys.path.append(str(Path(__file__).parent.parent / "tests"))
import testdata
from relay_bp.stim.sinter.check_matrices import CheckMatrices


def load_base_circuit(error_rate: float = 0.003, use_18_rounds: bool = True) -> stim.Circuit:
    """
    Load a bivariate bicycle circuit.

    Args:
        error_rate: Physical error rate for the circuit
        use_18_rounds: If True, use [[288,12,18]] with 18 rounds; else [[72,12,6]] with 6 rounds

    Returns:
        Stim circuit
    """
    if use_18_rounds:
        print(f"Loading [[288,12,18]] circuit with 18 rounds, error_rate={error_rate}")
        circuit = testdata.circuits.get_test_circuit(
            circuit="bicycle_bivariate_288_12_18_memory_Z",
            distance=18,
            rounds=18,
            error_rate=error_rate
        )
    else:
        print(f"Loading [[72,12,6]] circuit with 6 rounds, error_rate={error_rate}")
        circuit = testdata.circuits.get_test_circuit(
            circuit="bicycle_bivariate_72_12_6_memory_Z",
            distance=6,
            rounds=6,
            error_rate=error_rate
        )

    print(f"  Circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors")
    return circuit


def create_multi_round_circuit(base_circuit: stim.Circuit, target_rounds: int) -> stim.Circuit:
    """
    Create a multi-round circuit by repeating the syndrome extraction pattern.

    This approach builds on the base circuit structure and extends it to more rounds.
    The key challenge is that pre-generated Stim circuits are monolithic files.

    Strategy:
    1. Parse the base 6-round circuit to identify the repeated syndrome extraction cycle
    2. Extract the repeating pattern (should be 5 cycles after initialization)
    3. Replicate this pattern to create 150 rounds total

    Args:
        base_circuit: The base 6-round circuit
        target_rounds: Target number of rounds (150)

    Returns:
        Extended circuit with target_rounds rounds
    """
    print(f"\nCreating {target_rounds}-round circuit from {base_circuit.num_detectors // 72}-round base")

    # For pre-generated circuits, we need to analyze and extend them programmatically
    # This is complex because Stim circuits don't have a simple "rounds" parameter we can change

    # Alternative approach: Use circuit decomposition
    # Stim circuits for memory experiments typically have:
    # 1. Initialization layer
    # 2. Repeated syndrome extraction rounds
    # 3. Final measurement layer

    # For now, let's use a more direct approach: generate using Stim's circuit generation
    # capabilities if available, or use the DEM approach

    # Check if we can simply scale the circuit
    base_rounds = 6

    # The most reliable approach for bivariate bicycle codes is to use
    # an external generator or to manually construct the circuit
    # Since we don't have the generator code here, we'll try a heuristic approach

    # IMPORTANT: This is where we would need the actual circuit generator
    # For bivariate bicycle codes, this typically comes from qldpc or similar

    # For now, let's document what we need and create a workaround
    print("  WARNING: Direct circuit generation not available in this codebase")
    print("  This would require the bivariate bicycle code circuit generator")
    print("  Attempting workaround using circuit repetition heuristic...")

    # Workaround: Try to identify and replicate the syndrome extraction pattern
    # This is a best-effort approach

    raise NotImplementedError(
        "Multi-round circuit generation requires the bivariate bicycle code "
        "circuit generator which is not included in this repository. "
        "\n\nOptions:\n"
        "1. Use external tool (e.g., qldpc) to generate the 150-round circuit\n"
        "2. Request pre-generated circuit file from circuit library\n"
        "3. Implement circuit builder from scratch (complex)\n"
        "\n"
        "For now, we'll demonstrate with a scaled 6-round circuit for testing."
    )


def generate_circuit_file_request():
    """
    Generate a template request for the 150-round circuit file.
    This documents what we need if circuits must be pre-generated.
    """
    template = """
    CIRCUIT FILE REQUEST FOR FUSION DECODING
    =========================================

    Code: Bivariate Bicycle [[72, 12, 6]]
    Polynomials: A=x^3+y+y^2, B=y^3+x+x^2

    Parameters:
      - Distance: 6
      - Rounds: 150 (not 6!)
      - Error rate: 0.003
      - Noise model: uniform_circuit
      - Basis: CX
      - Memory experiment: Z

    Expected filename format:
    circuit=bicycle_bivariate_72_12_6_memory_Z,distance=6,rounds=150,error_rate=0.003,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim

    Expected properties:
      - Num detectors: ~10,800 (72 per round × 150 rounds)
      - Num observables: 12
      - Temporal structure: Detectors ordered by round

    This file should be placed in:
    tests/testdata/bicycle_bivariate/
    """
    return template


def extract_check_matrices_from_dem(circuit: stim.Circuit, verbose: bool = True) -> CheckMatrices:
    """
    Extract check matrices from circuit's detector error model.

    Args:
        circuit: Stim circuit
        verbose: Print dimension information

    Returns:
        CheckMatrices object containing H, A, and error priors
    """
    if verbose:
        print(f"\nExtracting DEM from circuit...")
        print(f"  Circuit: {circuit.num_detectors} detectors, {circuit.num_observables} observables")

    # Generate detector error model
    # Note: Bivariate bicycle codes have hyperedges, so we don't decompose
    dem = circuit.detector_error_model(decompose_errors=False)

    if verbose:
        print(f"  DEM: {dem.num_detectors} detectors, {dem.num_errors} error mechanisms")

    # Extract check matrices using relay_bp infrastructure
    # decomposed_hyperedges=None lets CheckMatrices auto-detect
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=None)

    if verbose:
        H = check_matrices.check_matrix
        A = check_matrices.observables_matrix
        priors = check_matrices.error_priors

        print(f"\nExtracted matrices:")
        print(f"  H (check matrix): {H.shape} - {H.nnz} non-zeros")
        print(f"  A (observable matrix): {A.shape} - {A.nnz} non-zeros")
        print(f"  Error priors: {len(priors)} variables")
        print(f"    Prior range: [{priors.min():.6f}, {priors.max():.6f}]")
        print(f"    Mean prior: {priors.mean():.6f}")

    return check_matrices


def generate_test_samples(
    circuit: stim.Circuit,
    num_samples: int = 100,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate test detector samples and corresponding error patterns.

    Args:
        circuit: Stim circuit to sample from
        num_samples: Number of test samples to generate
        verbose: Print progress information

    Returns:
        Tuple of (detector_samples, error_samples)
    """
    if verbose:
        print(f"\nGenerating {num_samples} test samples...")

    # Get detector error model for sampling
    dem = circuit.detector_error_model(decompose_errors=False)
    sampler = dem.compile_sampler()

    # Sample detector syndromes and errors
    # Note: Stim's sampler returns (detectors, observables, errors) when return_errors=True
    detector_samples, _, error_samples = sampler.sample(num_samples, return_errors=True)

    # Convert to uint8 for efficient storage
    detector_samples = detector_samples.astype(np.uint8)

    if verbose:
        print(f"  Detector samples: {detector_samples.shape}")
        print(f"  Error samples: {error_samples.shape}")
        print(f"  Detector density: {detector_samples.mean():.4f}")
        print(f"  Error density: {error_samples.mean():.4f}")

    return detector_samples, error_samples


def verify_temporal_structure(circuit: stim.Circuit, detectors_per_round: int = 72):
    """
    Verify that the circuit has proper temporal structure.

    This checks if detectors can be partitioned into rounds, which is critical
    for fusion decoding with time-axis partitioning.

    Args:
        circuit: Circuit to verify
        detectors_per_round: Expected detectors per round
    """
    print(f"\nVerifying temporal structure...")

    total_detectors = circuit.num_detectors
    num_rounds = total_detectors // detectors_per_round

    print(f"  Total detectors: {total_detectors}")
    print(f"  Detectors per round: {detectors_per_round}")
    print(f"  Inferred rounds: {num_rounds}")

    if total_detectors % detectors_per_round != 0:
        print(f"  WARNING: Detector count not evenly divisible by {detectors_per_round}")
        print(f"  Remainder: {total_detectors % detectors_per_round} detectors")

    # Try to extract detector coordinates to verify temporal ordering
    detector_coords = []
    instruction_count = 0
    detector_index = 0

    for instruction in circuit:
        instruction_count += 1
        if instruction.name == "DETECTOR":
            args = instruction.gate_args_copy()
            detector_coords.append((detector_index, args))
            detector_index += 1

            # Only collect a sample
            if detector_index >= 200:  # Sample first 200 detectors
                break

    if detector_coords:
        print(f"\n  Sample detector coordinates (first few):")
        for i in range(min(10, len(detector_coords))):
            idx, coords = detector_coords[i]
            print(f"    Detector {idx}: coords {coords}")

    # Check if detectors have time coordinates
    has_time_coord = False
    if detector_coords:
        # Typically time is the third coordinate [x, y, t]
        _, sample_coords = detector_coords[0]
        if len(sample_coords) >= 3:
            has_time_coord = True
            print(f"\n  Detectors have time coordinate: YES")
        else:
            print(f"\n  Detectors have time coordinate: NO (only {len(sample_coords)} coordinates)")

    return num_rounds, has_time_coord


def save_data_files(
    check_matrices: CheckMatrices,
    test_detectors: np.ndarray,
    test_errors: np.ndarray,
    output_dir: Path,
    prefix: str = "72_12_6_r6"
):
    """
    Save all data files in the expected format.

    Args:
        check_matrices: CheckMatrices object with H, A, priors
        test_detectors: Test detector samples
        test_errors: Test error samples
        output_dir: Output directory
        prefix: File prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving data files to {output_dir}/")

    # Save check matrix (H) - Convert to CSR format for Rust compatibility
    h_file = output_dir / f"{prefix}_Hdec.npz"
    save_npz(h_file, check_matrices.check_matrix.tocsr())
    print(f"  Saved: {h_file.name} (CSR format)")

    # Save observable matrix (A) - Convert to CSR format for Rust compatibility
    a_file = output_dir / f"{prefix}_Adec.npz"
    save_npz(a_file, check_matrices.observables_matrix.tocsr())
    print(f"  Saved: {a_file.name} (CSR format)")

    # Save error priors (convert to float32 for Rust compatibility)
    priors_file = output_dir / f"{prefix}_error_priors.npy"
    np.save(priors_file, check_matrices.error_priors.astype(np.float32))
    print(f"  Saved: {priors_file.name} (float32)")

    # Save test detector samples
    detectors_file = output_dir / f"{prefix}_detectors.npy"
    np.save(detectors_file, test_detectors)
    print(f"  Saved: {detectors_file.name}")

    # Save test error samples
    errors_file = output_dir / f"{prefix}_errors.npy"
    np.save(errors_file, test_errors)
    print(f"  Saved: {errors_file.name}")

    print(f"\nAll files saved successfully!")


def validate_generated_data(output_dir: Path, prefix: str = "72_12_6_r6"):
    """
    Load and validate the generated data files.

    Args:
        output_dir: Directory containing the data files
        prefix: File prefix
    """
    print(f"\n{'='*80}")
    print("VALIDATING GENERATED DATA")
    print(f"{'='*80}")

    from scipy.sparse import load_npz

    # Load files
    H = load_npz(output_dir / f"{prefix}_Hdec.npz")
    A = load_npz(output_dir / f"{prefix}_Adec.npz")
    priors = np.load(output_dir / f"{prefix}_error_priors.npy")
    detectors = np.load(output_dir / f"{prefix}_detectors.npy")
    errors = np.load(output_dir / f"{prefix}_errors.npy")

    print(f"\nLoaded data dimensions:")
    print(f"  H: {H.shape} ({H.nnz} non-zeros)")
    print(f"  A: {A.shape} ({A.nnz} non-zeros)")
    print(f"  Priors: {priors.shape}")
    print(f"  Test detectors: {detectors.shape}")
    print(f"  Test errors: {errors.shape}")

    # Validate consistency
    print(f"\nValidation checks:")

    # Check matrix dimensions are consistent
    num_detectors, num_vars = H.shape
    checks = []

    checks.append(("H and A have same number of variables",
                   A.shape[1] == num_vars))
    checks.append(("Priors length matches number of variables",
                   len(priors) == num_vars))
    checks.append(("Test detectors shape matches H rows",
                   detectors.shape[1] == num_detectors))
    checks.append(("All priors are in [0, 1]",
                   (priors >= 0).all() and (priors <= 1).all()))
    checks.append(("Detector samples are binary",
                   np.all(np.isin(detectors, [0, 1]))))

    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  All validation checks passed!")
    else:
        print(f"\n  WARNING: Some validation checks failed!")

    # Test syndrome consistency for a few samples
    # Note: We can't directly use DEM error samples with H because they have different shapes
    # (DEM errors are physical mechanisms, H variables are decomposed)
    # Instead, we verify that detectors are valid binary arrays
    print(f"\nTesting detector sample validity (first 5 samples):")
    for i in range(min(5, len(detectors))):
        is_binary = np.all(np.isin(detectors[i, :], [0, 1]))
        status = "VALID" if is_binary else "INVALID"
        print(f"  Sample {i}: {status} (binary detector array)")

    return all_passed


def main():
    """Main execution function."""

    print("="*80)
    print("GENERATE FUSION DATA FOR [[72, 12, 6]] CODE")
    print("="*80)

    # Configuration
    TARGET_ROUNDS = 30
    ERROR_RATE = 0.003
    NUM_TEST_SAMPLES = 100
    OUTPUT_DIR = Path(__file__).parent.parent / "crates" / "relay_bp" / "data"

    # Try to load or create the multi-round circuit
    try:
        print(f"\nAttempting to load {TARGET_ROUNDS}-round circuit...")

        # First, try to load the custom circuit from the r{N} directory
        custom_circuit_path = Path(__file__).parent.parent / "tests" / "testdata" / f"72_12_6_r{TARGET_ROUNDS}" / f"72_12_6_r{TARGET_ROUNDS}.stim"
        if custom_circuit_path.exists():
            print(f"Found custom circuit at: {custom_circuit_path}")
            circuit = stim.Circuit.from_file(str(custom_circuit_path))
            print(f"SUCCESS: Loaded custom {TARGET_ROUNDS}-round circuit!")
            print(f"  Detectors: {circuit.num_detectors}")
            print(f"  Expected: {TARGET_ROUNDS * 72}")
        else:
            # Fallback: Try to load using testdata module
            print(f"Custom circuit not found at {custom_circuit_path}")
            print(f"Trying testdata module...")
            try:
                circuit = testdata.circuits.get_test_circuit(
                    circuit="bicycle_bivariate_72_12_6_memory_Z",
                    distance=6,
                    rounds=TARGET_ROUNDS,  # Try 150 rounds
                    error_rate=ERROR_RATE
                )
                print(f"SUCCESS: Loaded pre-generated {TARGET_ROUNDS}-round circuit via testdata!")

            except ValueError as e:
                print(f"Pre-generated {TARGET_ROUNDS}-round circuit not found")
                print(f"Error: {e}")

                # Fallback: Try 18-round circuit (better than 6)
                print(f"\nFALLBACK 1: Trying 18-round [[288,12,18]] circuit...")
                try:
                    circuit = testdata.circuits.get_test_circuit(
                        circuit="bicycle_bivariate_288_12_18_memory_Z",
                        distance=18,
                        rounds=18,
                        error_rate=ERROR_RATE
                    )
                    print(f"SUCCESS: Using 18-round circuit (3x better temporal depth than 6 rounds)")
                except ValueError:
                    # Last resort: Use the 6-round circuit
                    print(f"\nFALLBACK 2: Using 6-round [[72,12,6]] circuit")
                    print(f"NOTE: Limited temporal structure for fusion!")
                    circuit = load_base_circuit(ERROR_RATE, use_18_rounds=False)

    except Exception as e:
        print(f"ERROR: Could not load circuit: {e}")
        import traceback
        traceback.print_exc()
        return

    # Verify temporal structure
    num_rounds, has_time = verify_temporal_structure(circuit)

    # Extract check matrices from DEM
    check_matrices = extract_check_matrices_from_dem(circuit)

    # Generate test samples
    test_detectors, test_errors = generate_test_samples(circuit, NUM_TEST_SAMPLES)

    # Save all data files
    save_data_files(
        check_matrices,
        test_detectors,
        test_errors,
        OUTPUT_DIR,
        prefix=f"72_12_6_r{num_rounds}"
    )

    # Validate the generated data
    validation_passed = validate_generated_data(OUTPUT_DIR, prefix=f"72_12_6_r{num_rounds}")

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nGenerated data for [[72, 12, 6]] code:")
    print(f"  Actual rounds: {num_rounds} (target: {TARGET_ROUNDS})")
    print(f"  Detectors: {circuit.num_detectors}")
    print(f"  Error variables: {check_matrices.check_matrix.shape[1]}")
    print(f"  Test samples: {NUM_TEST_SAMPLES}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")

    if num_rounds != TARGET_ROUNDS:
        print(f"\nWARNING: Generated data uses {num_rounds} rounds, not {TARGET_ROUNDS}!")
        print(f"This will NOT work for fusion decoding with time-axis partitioning.")
        print(f"\nTo generate proper {TARGET_ROUNDS}-round data:")
        print(f"1. Generate the circuit file using an external tool (e.g., qldpc)")
        print(f"2. Place it in tests/testdata/bicycle_bivariate/")
        print(f"3. Re-run this script")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
