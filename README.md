# Task 11: Correlation Analysis & Clustered Heatmaps (Data Analyst Track)

This project provides an automated data validation, hierarchical clustering, and correlation analysis pipeline to quantify linear and monotonic relationships among continuous features.

## Key Features & Deliverables
* **Defensive Data Validation:** Includes `validate_numeric_dataframe()` and `run_unit_tests()` for empty inputs and type checks.
* **Hierarchical Clustering:** Automatically sorts features via agglomerative linkage to reveal co-moving clusters.
* **Masked Diverging Heatmap:** Generates an annotated lower-triangle correlation heatmap (`artifacts/clustered_correlation_heatmap.png`).
* **Multicollinearity Audit:** Identifies feature pairs with $\vert{}r\vert{} > 0.85$ and saves findings to `artifacts/correlation_summary.json`.

## Quickstart & Execution
1. Clone this repository:
   ```bash
   git clone [https://github.com/shreedharchalmi/placemux_dailytask11da.git](https://github.com/shreedharchalmi/placemux_dailytask11da.git)
   cd placemux_dailytask11da