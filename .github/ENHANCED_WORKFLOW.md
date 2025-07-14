# Enhanced GitHub Actions Workflow with Package Verification

## What's New

The GitHub Actions workflows have been enhanced to include **automatic package verification** after publishing to TestPyPI. This ensures that:

1. ✅ The package is properly published and available
2. ✅ The package can be installed from TestPyPI
3. ✅ The package imports correctly
4. ✅ The package functionality works as expected

## How the Enhanced Workflow Works

### 1. Build and Publish (Same as before)
- Updates version with timestamp
- Builds the package
- Publishes to TestPyPI

### 2. **NEW**: Package Verification
- Waits 30 seconds for package availability on TestPyPI
- Creates a fresh virtual environment
- Installs the published package from TestPyPI
- Verifies the package can be imported
- Runs the test script (`src/tests/test_run.sh`)
- Checks that test output is generated
- Reports success or failure

## Test Script Details

The workflow runs `src/tests/test_run.sh` which:
- Uses the `pepe` command with test parameters
- Processes `src/tests/test_files/test.fasta`
- Uses the example model `examples/custom_model/example_protein_model`
- Generates output in `src/tests/test_files/test_output`
- Extracts various embedding types (per_token, mean_pooled, etc.)

## Workflow Output

You'll see detailed logs showing:
```
Package published to TestPyPI!
Waiting for package to be available on TestPyPI...
Installing package from TestPyPI...
Package imported successfully
Running test script...
✅ Test completed successfully - output directory created
✅ Package verification completed successfully!
```

## Benefits

1. **Confidence**: Know that your package actually works after publishing
2. **Early Detection**: Catch issues before users try to install
3. **End-to-end Testing**: Verifies the complete installation → usage pipeline
4. **Automatic**: No manual intervention needed

## Both Workflows Enhanced

- `publish-test-branch-trusted.yml` - Trusted publishing + verification

This single workflow handles everything you need for secure publishing and verification.

## Failure Handling

If any step fails (installation, import, or test execution), the workflow will:
- Show detailed error logs
- Mark the workflow as failed
- Prevent false positives where publishing succeeds but package is broken

## Test Requirements

The verification works because your repository includes:
- ✅ Test script: `src/tests/test_run.sh`
- ✅ Test data: `src/tests/test_files/test.fasta`
- ✅ Test config: `src/tests/test_files/test_substring.csv`
- ✅ Example model: `examples/custom_model/example_protein_model/`

This gives you complete confidence that your TestPyPI releases are fully functional!
