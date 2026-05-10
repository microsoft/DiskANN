# DiskAnnPy

Python bindings for DiskANN, providing efficient vector quantization and approximate nearest neighbor search.

## Installation

### Prerequisites

- Python 3.9+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

### Building from Source

Navigate to `diskann/DiskAnnPy`:

#### Linux

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install build dependencies
pip install maturin numpy

# Option 1: Build wheel (creates .whl in DiskANN/target/wheels)
maturin build -r

# Option 2: Build and install directly into active environment
maturin develop -r
```

#### Windows

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install build dependencies
pip install maturin numpy

# Option 1: Build wheel (creates .whl in DiskANN\target\wheels)
maturin build -r

# Option 2: Build and install directly into active environment
maturin develop -r
```

### Running Tests

Once installed, run the Python tests from the `DiskAnnPy` directory:

```bash
python -m unittest discover tests
```

## Quantizers

See [src/quantization/README.md](src/quantization/README.md) for documentation on the vector quantization APIs.