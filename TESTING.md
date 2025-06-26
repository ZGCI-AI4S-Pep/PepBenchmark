# 🧪 Testing Guide for PepBenchmark

## Overview

This project includes comprehensive automated testing integrated with pre-commit hooks to ensure code quality and reliability.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── test_metadata.py              # Metadata functionality tests
├── test_dataset_loader/          # Dataset loading tests
│   ├── test_base_dataset.py
│   └── test_singlepep_pred_dataset.py
├── test_pep_utils/               # Peptide utilities tests
│   ├── test_convert.py
│   └── test_featurizer.py
├── test_utils/                   # General utilities tests
│   └── test_split.py
└── test_visualization/           # Visualization tests
```

## Running Tests

### 🚀 Quick Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html

# Run only fast tests (exclude slow integration tests)
python -m pytest tests/ -m "not slow" -v

# Run specific module tests
python -m pytest tests/test_dataset_loader/ -v
python -m pytest tests/test_pep_utils/ -v
python -m pytest tests/test_utils/ -v

# Run single test file
python -m pytest tests/test_metadata.py -v

# Use the convenience script
./run_tests.sh
```

### 🔧 Test Configuration

- **pytest.ini**: Main pytest configuration
- **pytest-precommit.ini**: Simplified config for pre-commit hooks
- **requirements-test.txt**: Testing dependencies

## Pre-commit Integration

### 🛡️ Automatic Checks

The pre-commit system runs the following checks:

#### On Commit (`pre-commit`):
1. **Code Quality**:
   - Remove trailing whitespace
   - Check Python AST validity
   - Detect merge conflicts
   - Insert license headers
   - Run flake8 linting
   - Run Ruff linting and formatting

2. **Basic Validation**:
   - Python syntax check
   - Basic import validation
   - Quick validation script

#### On Push (`pre-push`):
1. **Full Test Suite**:
   - Run all unit tests
   - Check for test failures
   - Ensure core functionality works

### 🎯 Targeted Testing

The system intelligently runs different tests based on what files changed:

- **Core modules changed** → Run targeted tests
- **Dataset loader changed** → Run dataset tests
- **Pep utils changed** → Run pep utils tests
- **Any Python files** → Run syntax and import checks

### ⚙️ Setup Pre-commit

```bash
# Install pre-commit hooks
pre-commit install

# Install pre-push hooks
pre-commit install --hook-type pre-push

# Test pre-commit without committing
pre-commit run --all-files

# Run specific hook
pre-commit run basic-validation
pre-commit run pytest-quick
```

## Test Categories

### 🏃‍♂️ Fast Tests
- Unit tests for individual functions
- Mock-based tests
- Quick validation checks
- **Marked with**: No special marker (default)

### 🐌 Slow Tests
- Integration tests
- Tests requiring external dependencies
- End-to-end workflow tests
- **Marked with**: `@pytest.mark.slow`

### 🧩 Integration Tests
- Cross-module functionality tests
- Full pipeline tests
- **Marked with**: `@pytest.mark.integration`

## Writing Tests

### 📝 Test Naming Convention

```python
class TestClassName:
    def test_method_name_condition(self):
        """Test description of what is being tested."""
        # Arrange
        # Act
        # Assert
```

### 🛠️ Using Fixtures

```python
def test_with_sample_data(sample_peptide_data):
    """Test using shared sample data fixture."""
    assert len(sample_peptide_data) > 0
```

### 🎭 Mocking External Dependencies

```python
@patch('pepbenchmark.pep_utils.convert.Chem')
def test_with_mock(self, mock_chem):
    """Test with mocked RDKit dependency."""
    mock_chem.MolFromSequence.return_value = MagicMock()
    # Test code here
```

## Continuous Integration

The testing system is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    python -m pytest tests/ --cov=src/pepbenchmark --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Maintenance

### 🔍 Debugging Failed Tests

```bash
# Run with verbose output and stop on first failure
python -m pytest tests/ -v -x --tb=long

# Run specific failing test
python -m pytest tests/test_module.py::TestClass::test_method -v

# Run with debugger
python -m pytest tests/ --pdb
```

### 📊 Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html
# Open htmlcov/index.html in browser

# Generate terminal coverage report
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=term-missing
```

### 🧹 Test Cleanup

- Mock external dependencies appropriately
- Use temporary files/directories
- Clean up after tests complete
- Use fixtures for shared setup/teardown

## Best Practices

1. **Write tests first** (TDD approach when possible)
2. **Keep tests fast** - use mocks for external dependencies
3. **Test edge cases** - empty inputs, invalid data, etc.
4. **Use descriptive test names** - explain what's being tested
5. **One assertion per test** when possible
6. **Use fixtures** for common test data
7. **Mark slow tests** appropriately

## Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and module structure
2. **Missing dependencies**: Install test requirements
3. **Slow tests**: Use mocks instead of real external calls
4. **Flaky tests**: Check for race conditions or external dependencies

### Getting Help

- Check test output for specific error messages
- Look at similar working tests for examples
- Use `pytest --collect-only` to see test discovery
- Run with `-v` flag for verbose output
