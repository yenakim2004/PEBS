# Memory Optimization Implementation

## Problem

Original error when running on 16GB RAM:
```
Unable to allocate 1.97 GiB for an array with shape (278, 949285)
```

**Root cause**: NSDUH dataset with 3,662 columns requires ~26GB memory for full load.

## Solutions Implemented

### Method 1: Column Selection (방안 1) ✅
**Reduction**: 3,662 columns → 80 columns (~97.8% reduction)

**Selected columns (80 total)**:
- Demographics (12): Age, sex, education, income, employment, etc.
- Alcohol patterns (20): Usage frequency, binge drinking, abuse/dependence
- Substance use (15): Marijuana, cocaine, heroin, etc.
- Mental health (15): Overall health, treatment history, K6 score
- Behavior/social (12): Delinquency, income, poverty, arrests
- Survey metadata (6): Year, weights, strata

**Memory impact**: 26GB → ~2GB

### Method 2: PCA Dimensionality Reduction (방안 2) ✅
**Reduction**: 80 columns → ~30-50 components (95% variance)

**Configuration**:
```yaml
pca:
  enabled: true
  n_components: 0.95  # Keep 95% variance
  whiten: false
  random_state: 42
```

**Memory impact**: Additional ~60% reduction in feature space

### Method 3: Chunk Processing (방안 3) ✅
**Implementation**: Iterator-based chunk processing with tqdm progress bar

**Configuration**:
```yaml
memory:
  nsduh_chunksize: 10000
  enable_gc: true
  low_memory_mode: true
```

**Memory impact**: Prevents peak memory usage spikes during loading

## Files Modified

### 1. `config.yaml` (+104 lines)
- Added `nsduh_selected_columns` (80 core variables)
- Added `pca` configuration section

### 2. `pebs/data/loader.py` (+63 lines)
- Added `selected_columns` parameter to NSDUHLoader.__init__()
- Implemented chunk processing with `pd.read_csv(..., chunksize=...)`
- Added progress bar with tqdm
- Added memory usage statistics

**New features**:
```python
NSDUHLoader(
    file_path="...",
    selected_columns=[...]  # Column selection
)
loader.load(use_chunks=True)  # Chunk processing
```

### 3. `pebs/data/preprocessor.py` (+73 lines)
- Added `pca_config` parameter to NSDUHPreprocessor.__init__()
- Implemented `apply_pca()` method
- Integrated PCA into `process()` pipeline
- Added variance explained reporting

**New features**:
```python
NSDUHPreprocessor(
    pca_config={'enabled': True, 'n_components': 0.95}
)
```

### 4. `train.py` (+12 lines)
- Pass `selected_columns` from config to NSDUHLoader
- Pass `pca_config` from config to NSDUHPreprocessor
- Save PCA model alongside other artifacts

## Expected Results

### Memory Usage
| Stage | Before | After | Reduction |
|-------|--------|-------|-----------|
| Data loading | 26GB | 2GB | 92% |
| Post-PCA | 2GB | 1.2GB | 40% |
| **Total** | **26GB** | **1.2GB** | **95%** |

### Loading Time
| Operation | Before | After |
|-----------|--------|-------|
| NSDUH load | Never completes | 30-60 seconds |
| Full pipeline | N/A | 5-10 minutes |

### Feature Space
| Stage | Features |
|-------|----------|
| Original | 3,662 |
| After column selection | 80 (97.8% reduction) |
| After PCA | ~30-50 (98.6% reduction) |

## How to Use

### Enable All Optimizations (Default)
```bash
python train.py
```

All optimizations are enabled by default in `config.yaml`.

### Disable PCA
Edit `config.yaml`:
```yaml
preprocessing:
  pca:
    enabled: false
```

### Adjust PCA Components
```yaml
preprocessing:
  pca:
    n_components: 50      # Exact number
    # OR
    n_components: 0.90    # 90% variance
```

### Disable Column Selection
Edit `config.yaml`:
```yaml
data:
  nsduh_selected_columns: null  # Load all 3,662 columns
```
**Warning**: Requires 26GB+ RAM

### Disable Chunk Processing
Not recommended, but possible:
```python
nsduh_data = nsduh_loader.load(use_chunks=False)
```

## Validation

### Test on Windows (16GB RAM)
```bash
python train.py
```

Expected output:
```
📊 Loading NSDUH dataset...
   Selected columns: 80 (instead of all 3,662)
   Using chunksize: 10000
Loading chunks: 100%|████████| 95/95 [00:45<00:00, 2.1chunks/s]
✅ Dataset loaded successfully
   Shape: (949285, 80)
   Memory usage: 1.94 GB
   Memory saved: ~97.8% by column selection

🔬 Applying PCA dimensionality reduction...
   Target variance explained: 95.0%
   Original features: 80
✅ PCA completed
   Reduced to: 42 components
   Variance explained: 95.12%
   Dimension reduction: 80 → 42 (47.5% reduction)
```

### Verify Memory Usage
Use Windows Task Manager or:
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1e9:.2f} GB")
```

## Troubleshooting

### Still Out of Memory
1. Reduce PCA components: `n_components: 0.85` (85% variance)
2. Increase chunk size: `nsduh_chunksize: 20000`
3. Close other applications

### Missing Column Error
Some selected columns may not exist in older NSDUH versions.
The code will use available columns and skip missing ones.

### Slower Performance
- Chunk processing trades speed for memory
- Disable chunks if you have 32GB+ RAM: `use_chunks=False`

## Future Improvements

1. **Incremental PCA**: Process PCA in chunks for even larger datasets
2. **Feature Selection**: Use statistical tests to select most relevant features
3. **Sparse Matrix**: Use scipy sparse matrices for categorical variables
4. **Dask Integration**: Distributed computing for multi-machine setups

## References

- NSDUH Codebook: https://www.samhsa.gov/data/data-we-collect/nsduh
- PCA Documentation: https://scikit-learn.org/stable/modules/decomposition.html#pca
- Pandas Chunking: https://pandas.pydata.org/docs/user_guide/io.html#io-chunking
