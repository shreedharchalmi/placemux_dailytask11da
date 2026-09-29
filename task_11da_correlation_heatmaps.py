import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("artifacts/task11da_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Task11DA")

# ---------------------------------------------------------
# Defensive Data Validation & Unit Tests
# ---------------------------------------------------------
def validate_numeric_dataframe(df: pd.DataFrame) -> None:
    """Validates dataframe integrity, non-emptiness, and numeric typing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")
    
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns detected: {non_numeric}")
        
    zero_var = [col for col in df.columns if df[col].nunique() <= 1]
    if zero_var:
        raise ValueError(f"Zero-variance columns found: {zero_var}")
    logger.info("DataFrame validation passed: all columns are continuous/numeric.")

def run_unit_tests():
    """Executes automated unit tests exercising edge cases."""
    logger.info("Running automated edge-case unit tests...")
    
    # Test 1: Empty DataFrame
    try:
        validate_numeric_dataframe(pd.DataFrame())
    except ValueError as e:
        assert "cannot be empty" in str(e)
        logger.info("Unit Test 1 Passed: Empty DataFrame caught.")

    # Test 2: Non-numeric DataFrame
    try:
        mock_df = pd.DataFrame({"num": [1, 2, 3], "str": ["a", "b", "c"]})
        validate_numeric_dataframe(mock_df)
    except ValueError as e:
        assert "Non-numeric columns detected" in str(e)
        logger.info("Unit Test 2 Passed: Non-numeric columns caught.")

    logger.info("All unit tests passed successfully.\n")

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    run_unit_tests()
    logger.info("Starting Task 11: Correlation & Heatmaps Analysis Pipeline")
    
    # 1. Ingestion
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    validate_numeric_dataframe(df)
    
    # Select key numerical features across mean and worst categories for clear visualization
    key_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst concavity', 'worst concave points', 'target'
    ]
    df_sub = df[key_features]
    
    # 2. Compute Pearson & Spearman Correlation Matrices
    corr_pearson = df_sub.corr(method='pearson')
    corr_spearman = df_sub.corr(method='spearman')
    logger.info("Computed Pearson and Spearman correlation matrices.")
    
    # 3. Hierarchical Clustering to Group Co-Moving Variables
    # Convert correlation to distance metric: d = 1 - |r|
    dist = pdist(corr_pearson.values, metric='euclidean')
    linkage = hierarchy.linkage(dist, method='average')
    order = hierarchy.leaves_list(linkage)
    
    clustered_features = [key_features[i] for i in order]
    corr_clustered = corr_pearson.loc[clustered_features, clustered_features]
    logger.info("Reordered correlation matrix using hierarchical clustering.")
    
    # 4. Generate Masked Clustered Heatmap
    mask = np.triu(np.ones_like(corr_clustered, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_clustered,
        mask=mask,
        cmap='coolwarm',
        vmin=-1.0,
        vmax=1.0,
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8}
    )
    plt.title("Clustered Correlation Heatmap (Key Morphological Signals)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plot_path = "artifacts/clustered_correlation_heatmap.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved clustered correlation heatmap to {plot_path}")
    
    # 5. Extract Multicollinear Pairs (|r| > 0.85)
    corr_matrix = df_sub.drop(columns=['target']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    multicollinear_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val > 0.85:
                multicollinear_pairs.append({
                    "feature_1": idx,
                    "feature_2": col,
                    "pearson_r": round(float(val), 4)
                })
                
    # Extract top target drivers (correlations with binary target)
    target_corrs = corr_pearson['target'].drop('target').sort_values().to_dict()
    
    logger.info(f"Identified {len(multicollinear_pairs)} feature pairs with severe multicollinearity (|r| > 0.85).")
    
    # 6. Save JSON Summary Artifact
    summary = {
        "task": "Task 11 - Correlation & Heatmaps",
        "sample_count": len(df_sub),
        "analyzed_features": key_features,
        "severe_multicollinear_pairs_gt_0_85": multicollinear_pairs,
        "correlations_with_target": {k: round(v, 4) for k, v in target_corrs.items()}
    }
    
    summary_path = "artifacts/correlation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved correlation summary JSON to {summary_path}")
    logger.info("Task 11 execution completed successfully.")

if __name__ == "__main__":
    main()