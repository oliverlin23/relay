# Generating Test Data for Different Round Numbers

This guide explains how to generate test data for the fusion decoder benchmarks with different numbers of rounds.

## Overview

The benchmark uses test data files for the [[72, 12, 6]] bivariate bicycle code. The data includes:
- Check matrix (H) and observable matrix (A) in sparse format
- Error priors
- Test detector samples and corresponding errors

The data files are named with the pattern: `72_12_6_r{N}_*` where `{N}` is the number of rounds.

## Current Configuration

The benchmark (`benchmark_time_axis_partitioning.rs`) is currently configured to use **6 rounds** by default:
- Data directory: `tests/testdata/72_12_6_r6/`
- Data files: `crates/relay_bp/data/72_12_6_r6_*`

## Method 1: Using Pre-generated Circuits (Recommended)

If you have a pre-generated Stim circuit file for the desired number of rounds:

### Step 1: Prepare the Circuit File

1. Generate or obtain a Stim circuit file for the [[72, 12, 6]] code with your target number of rounds
2. The circuit should be a memory experiment (Z basis) with the bivariate bicycle code
3. Expected filename format: `72_12_6_r{N}.stim` where `{N}` is the number of rounds

### Step 2: Place the Circuit File

Place the circuit file in the testdata directory:
```bash
tests/testdata/72_12_6_r{N}/72_12_6_r{N}.stim
```

For example, for 150 rounds:
```bash
tests/testdata/72_12_6_r150/72_12_6_r150.stim
```

### Step 3: Generate the Data Files

Modify `examples/generate_fusion_data.py` to set your target number of rounds:

```python
# In the main() function, change:
TARGET_ROUNDS = 150  # Change to your desired number of rounds
```

Then run the generation script:

```bash
cd /path/to/relay
python examples/generate_fusion_data.py
```

The script will:
1. Load the circuit from `tests/testdata/72_12_6_r{N}/72_12_6_r{N}.stim`
2. Extract check matrices and error priors
3. Generate test samples
4. Save all data files to `crates/relay_bp/data/72_12_6_r{N}_*`

### Step 4: Update the Benchmark

Update `crates/relay_bp/examples/benchmark_time_axis_partitioning.rs` to use your new data:

```rust
// Change these lines:
let code_path = resources.join("72_12_6_r{N}");  // Replace {N} with your round number
let test_detectors: Array2<Bit> =
    read_npy(resources.join("72_12_6_r{N}_detectors.npy"))  // Replace {N}
        .expect("Failed to load test detectors");
```

## Method 2: Using the Testdata Module

If the testdata module has pre-generated circuits for your desired number of rounds:

### Step 1: Check Available Circuits

The testdata module may have circuits available through:
```python
from tests import testdata

# Check what's available
circuit = testdata.circuits.get_test_circuit(
    circuit="bicycle_bivariate_72_12_6_memory_Z",
    distance=6,
    rounds=YOUR_ROUND_NUMBER,  # Try your desired number
    error_rate=0.003
)
```

### Step 2: Generate Data

Modify `examples/generate_fusion_data.py`:

```python
TARGET_ROUNDS = YOUR_ROUND_NUMBER  # e.g., 18, 30, 50, etc.
```

The script will attempt to load the circuit via the testdata module first, then fall back to other methods.

## Method 3: Generating Circuits Externally

If you need to generate circuits for round numbers not available in the testdata:

### Using qldpc or Similar Tools

1. Use an external circuit generator (e.g., qldpc) to create the Stim circuit file
2. Ensure the circuit has:
   - Code: [[72, 12, 6]] bivariate bicycle
   - Polynomials: A=x³+y+y², B=y³+x+x²
   - Memory experiment: Z basis
   - Error rate: 0.003 (or your desired rate)
   - Noise model: uniform_circuit
   - Basis: CX

3. Save the circuit as `tests/testdata/72_12_6_r{N}/72_12_6_r{N}.stim`
4. Follow Method 1, Step 3 onwards

## Expected File Structure

After generation, you should have these files in `crates/relay_bp/data/`:

```
72_12_6_r{N}_Hdec.npz          # Check matrix (sparse CSR format)
72_12_6_r{N}_Adec.npz          # Observable matrix (sparse CSR format)
72_12_6_r{N}_error_priors.npy  # Error priors (1D array)
72_12_6_r{N}_detectors.npy     # Test detector samples (2D array)
72_12_6_r{N}_errors.npy        # Corresponding error patterns (2D array)
```

And in `tests/testdata/72_12_6_r{N}/`:

```
72_12_6_r{N}.stim              # Stim circuit file
72_12_6_r{N}.dem               # Detector error model
72_12_6_r{N}.dem.b8            # Binary DEM format
72_12_6_r{N}.obs.b8            # Binary observables format
```

## Verification

After generating data, verify it's correct:

```bash
# Run the benchmark
cargo run --example benchmark_time_axis_partitioning --release
```

The benchmark will:
1. Load the data files
2. Verify the temporal structure (detectors per round)
3. Run partition quality analysis
4. Validate correctness

## Common Round Numbers

- **6 rounds**: Default, minimal temporal structure
- **18 rounds**: Good for testing, available in testdata
- **30 rounds**: Moderate temporal depth
- **50 rounds**: Better for fusion partitioning
- **100 rounds**: Good temporal depth
- **150 rounds**: High temporal depth, good for fusion demonstrations

## Troubleshooting

### "Failed to load quantum code"

- Check that the directory `tests/testdata/72_12_6_r{N}/` exists
- Verify the `.stim` file is present and valid
- Ensure the circuit has the expected number of detectors (72 × N)

### "Failed to load test detectors"

- Run the generation script to create the `.npy` files
- Check that files are in `crates/relay_bp/data/`
- Verify file naming matches the pattern `72_12_6_r{N}_*`

### Wrong number of rounds detected

- The benchmark calculates rounds as `total_detectors / 72`
- If this doesn't match your expected number, check:
  - The circuit file has the correct number of detectors
  - The circuit structure is correct (72 detectors per round)

## Example: Switching from 6 to 150 Rounds

1. **Ensure 150-round circuit exists**:
   ```bash
   ls tests/testdata/72_12_6_r150/72_12_6_r150.stim
   ```

2. **Generate data files**:
   ```bash
   # Edit examples/generate_fusion_data.py: TARGET_ROUNDS = 150
   python examples/generate_fusion_data.py
   ```

3. **Update benchmark**:
   ```rust
   // In benchmark_time_axis_partitioning.rs
   let code_path = resources.join("72_12_6_r150");
   read_npy(resources.join("72_12_6_r150_detectors.npy"))
   ```

4. **Run benchmark**:
   ```bash
   cargo run --example benchmark_time_axis_partitioning --release
   ```

## Notes

- The generation script automatically detects the actual number of rounds from the circuit
- If the target rounds don't match the circuit, a warning will be displayed
- The script validates temporal structure to ensure proper round ordering
- Test samples are generated with the same error rate as the circuit (default: 0.003)

