# Column Similarity Alignment Method

## Overview

The **column similarity** alignment method is a new approach for aligning clusters between acoustic and perceptual modalities in the NMF clustering pipeline. This method directly compares the coefficient matrices (W matrices) from NMF decomposition of each modality.

## How It Works

### Step 1: Extract Coefficient Matrices
- After NMF decomposition, extract coefficient matrices W_A (acoustic) and W_B (perceptual)
- These matrices have shape (n_patients, k_components) where each row represents a patient's membership weights across components

### Step 2: Row Normalization
- Normalize each patient's membership weights so they sum to 1
- This ensures fair comparison across different scales and modalities
- Formula: `W_norm[i, :] = W[i, :] / sum(W[i, :])`

### Step 3: Compute Similarity Matrix
- Create a k_A × k_B similarity matrix C
- C[i,j] = similarity between acoustic component i and perceptual component j
- Computed across all shared patients using their normalized membership weights

### Step 4: Apply Optimal Mapping (if k_A == k_B)
- Use Hungarian algorithm on -C to find optimal 1-to-1 mapping
- This maximizes total similarity across all component pairs
- If k_A ≠ k_B, no optimal mapping is possible, but similarity matrix is still informative

## Configuration

Add these parameters to your `src/config.py`:

```python
# Cross-modal alignment method configuration
ALIGNMENT_METHOD = "column_similarity"  # Options: "jsd_hungarian", "column_similarity"

# Column similarity alignment parameters
COLUMN_SIMILARITY_METRIC = "cosine"  # Options: "cosine", "correlation"
```

## Similarity Metrics

### Cosine Similarity
- Range: [0, 1]
- Higher values indicate more similar components
- Good for capturing overall pattern similarity
- Less sensitive to magnitude differences

### Correlation (Pearson)
- Range: [-1, 1]
- Values close to ±1 indicate strong linear relationship
- Can capture both positive and negative associations
- More sensitive to linear patterns

## Output Files

When using column similarity, the following additional files are generated:

### Results Folder
- `similarity_matrix_{metric}.csv` - Full k_A × k_B similarity matrix
- `cluster_mapping_{metric}.csv` - Optimal cluster mapping with similarity scores
- `similarity_matrix_metadata_{metric}.json` - Analysis metadata and parameters

### Plots Folder
- `similarity_matrix_heatmap_{metric}.png` - Visual heatmap of similarity matrix

## Interpreting Results

### Similarity Matrix Heatmap
- **Rows**: Acoustic components (A0, A1, A2, ...)
- **Columns**: Perceptual components (P0, P1, P2, ...)
- **Color Intensity**: Similarity strength
- **Red Borders**: Optimal mapping when k_A == k_B

### Quick Inspection Tips
1. **Diagonal Patterns**: Look for high similarity values that suggest natural alignments
2. **Optimal Mappings**: Check if Hungarian algorithm results make intuitive sense
3. **Similarity Distribution**: Examine overall similarity patterns across the matrix
4. **Metric Comparison**: Compare results between cosine and correlation metrics

### Example Interpretation
```
Similarity Matrix (Cosine):
       P0    P1    P2
A0   0.85  0.23  0.41  <- A0 most similar to P0
A1   0.34  0.78  0.19  <- A1 most similar to P1  
A2   0.12  0.31  0.89  <- A2 most similar to P2

Optimal Mapping: A0→P0, A1→P1, A2→P2
Average Similarity: 0.84
```

## Advantages

1. **Direct Comparison**: Compares actual NMF coefficient matrices rather than derived prototypes
2. **Patient-Level Information**: Uses all patient membership data, not just cluster centroids
3. **Flexible Metrics**: Supports multiple similarity measures (cosine, correlation)
4. **Visual Interpretation**: Clear heatmap visualization for easy inspection
5. **Preserves Uncertainty**: Accounts for soft cluster memberships rather than hard assignments

## When to Use

- **Equal Components**: Most effective when k_A == k_B for optimal 1-to-1 mapping
- **Soft Clustering**: When you want to leverage continuous membership weights
- **Pattern Discovery**: When looking for subtle cross-modal relationships
- **Method Comparison**: As an alternative to JSD+Hungarian for validation

## Comparison with JSD+Hungarian

| Aspect | Column Similarity | JSD+Hungarian |
|--------|------------------|---------------|
| Input | Raw coefficient matrices | Cluster prototypes |
| Information | All patient data | Averaged cluster centers |
| Mapping | Direct coefficient comparison | Prototype divergence |
| Flexibility | Multiple similarity metrics | Fixed JSD metric |
| Visualization | Similarity heatmap | Confusion matrix |

## Usage Example

```python
# 1. Configure method in config.py
ALIGNMENT_METHOD = "column_similarity"
COLUMN_SIMILARITY_METRIC = "cosine"

# 2. Run analysis
python run_analysis.py

# 3. Check outputs
# - results/similarity_matrix_cosine.csv
# - results/cluster_mapping_cosine.csv  
# - plots/similarity_matrix_heatmap_cosine.png
```

## Mathematical Details

### Row Normalization
For patient i: `W_norm[i, j] = W[i, j] / Σ_k W[i, k]`

### Cosine Similarity
`cosine(A_i, P_j) = (A_i · P_j) / (||A_i|| × ||P_j||)`

### Correlation
`corr(A_i, P_j) = cov(A_i, P_j) / (σ_A_i × σ_P_j)`

### Hungarian Algorithm
Solves: `min Σ c[i,π(i)]` where π is permutation
For similarity maximization: use cost = -similarity

## Notes

- Method automatically handles cases where k_A ≠ k_B
- Adds small epsilon (1e-10) to avoid division by zero in normalization
- NaN values in correlation are set to 0.0
- Results are saved with metric suffix for easy identification
- Existing JSD+Hungarian method remains unchanged and available
