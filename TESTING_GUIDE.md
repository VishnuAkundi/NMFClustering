# Testing Guide for ALS Clustering Analysis Pipeline

## Overview
This guide provides step-by-step instructions for testing your modular ALS clustering analysis pipeline.

## Quick Testing Steps

### 1. **Quick Dependency Check**
First, verify all required packages are installed:

```powershell
# Navigate to your project directory
cd "C:\Users\akundivi\Desktop\Acoustic_Pipeline_New\VirtualSLP-analytics\Clustering\backup"

# Run quick test
python test_pipeline.py --quick
```

This will check:
- ✅ All Python packages are installed
- ✅ All custom modules can be imported  
- ✅ Required data files exist

### 2. **Full Pipeline Test**
If the quick test passes, run the full test:

```powershell
python test_pipeline.py
```

This performs a complete end-to-end test of all components.

## Alternative Testing Options

### Option A: Test with Your Real Data

1. **Run the pipeline directly:**
   ```powershell
   python run_analysis.py
   ```

2. **Check the outputs:**
   - Look for the `outputs/` directory
   - Verify plots are generated in `outputs/plots/`
   - Check tables in `outputs/tables/`
   - Review results in `outputs/results/`

### Option B: Test with Minimal Test Data

If you want to test without using your full dataset:

1. **Create test data:**
   ```powershell
   python create_test_data.py
   ```

2. **Backup your original config:**
   ```powershell
   mv src\config.py src\config_original.py
   mv src\config_test.py src\config.py
   ```

3. **Run tests:**
   ```powershell
   python test_pipeline.py
   ```

4. **Restore original config:**
   ```powershell
   mv src\config.py src\config_test.py
   mv src\config_original.py src\config.py
   ```

## Manual Testing Steps

### 1. **Test Data Loading**
```powershell
python -c "from src.data_loader import merge_all_data; print('Testing...'); data = merge_all_data(); print(f'Loaded {data[0].shape[0]} participants')"
```

### 2. **Test NMF Clustering**
```powershell
python -c "from src.nmf_clustering import run_nmf_clustering; print('NMF test - this may take a few minutes...')"
```

### 3. **Test Visualization**
```powershell
python -c "from src.visualization import setup_output_directories; setup_output_directories(); print('Output directories created')"
```

## Troubleshooting Common Issues

### Issue: Import Errors
**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue: "No module named 'src'"
**Solution:** Make sure you're in the correct directory:
```powershell
cd "C:\Users\akundivi\Desktop\Acoustic_Pipeline_New\VirtualSLP-analytics\Clustering\backup"
```

### Issue: Data files not found
**Solution:** Check your config.py file paths:
```python
# In src/config.py, verify these paths point to your actual data:
ACOUSTIC_FEATURES_PATH = "../AllFeats/*_converted.csv"
PERCEPTUAL_DATA_PATH = "../feature_weights_perceptual.csv"
DEMOGRAPHICS_PATH = "../Clustering_all_results_combined.csv"
```

### Issue: Memory errors during NMF
**Solution:** Reduce the number of components or runs in config.py:
```python
N_COMPONENTS_RANGE = [2, 3, 4]  # Instead of larger range
N_RUNS = 5  # Instead of 10
```

## Performance Testing

### Timing Test
```powershell
python -c "import time; start=time.time(); exec(open('run_analysis.py').read()); print(f'Total time: {time.time()-start:.2f} seconds')"
```

### Memory Usage Test
```powershell
# Install psutil if not available: pip install psutil
python -c "import psutil, os; process = psutil.Process(os.getpid()); print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')"
```

## Expected Outputs

After successful testing, you should see:

### Files Created:
```
outputs/
├── plots/
│   ├── pca_acoustic_clusters_*.png
│   ├── pca_perceptual_clusters_*.png
│   ├── basis_distributions_*.png
│   └── coefficient_distributions_*.png
├── tables/
│   ├── cluster_assignments_*.csv
│   ├── top_features_*.csv
│   ├── demographic_stats.csv
│   └── cross_modal_analysis_*.csv
└── results/
    └── analysis_summary_*.txt
```

### Console Output:
```
✅ Data loaded successfully
✅ NMF clustering completed
✅ Visualizations generated
✅ Cross-modal analysis completed
✅ Results exported
🎉 Analysis complete!
```

## Next Steps After Testing

1. **Review the results** in the `outputs/` directory
2. **Check the analysis summary** for key findings
3. **Validate the cluster assignments** make sense for your data
4. **Adjust parameters** in `config.py` if needed
5. **Run feature extraction** on new audio files if needed

## Getting Help

If tests fail:
1. Check the error messages carefully
2. Verify all file paths in `config.py`
3. Ensure all dependencies are installed
4. Try the minimal test data approach
5. Check that your data files have the expected format

## Production Checklist

Before using in production:
- [ ] Quick test passes
- [ ] Full test passes  
- [ ] All expected outputs are generated
- [ ] Results look reasonable
- [ ] Performance is acceptable
- [ ] Error handling works correctly
