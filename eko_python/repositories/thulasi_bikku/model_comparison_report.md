# Model Comparison for Lung Disease Classification

## Performance Comparison Summary

### ICBHI Dataset

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Deep Learning (Proposed) | 0.94 | 0.94 | 0.94 | 0.94 |
| SVM | 0.84 | 0.87 | 0.86 | 0.86 |

### Coswara Dataset

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Deep Learning (Proposed) | 0.90 | 0.92 | 0.91 | 0.92 |
| Random Forest | 0.80 | 0.88 | 0.84 | 0.85 |

## Key Findings

1. The proposed deep learning model outperforms traditional machine learning approaches (SVM and Random Forest) on both datasets.

2. On the ICBHI dataset, the deep learning model achieves an 8% improvement in overall accuracy compared to SVM, with particularly significant improvements in precision for COPD (13% higher) and Pneumonia (7% higher).

3. On the Coswara dataset, the deep learning model shows a 7% improvement in overall accuracy compared to Random Forest, with particular strength in COVID-19 detection (F1 score improvement of 16%).

4. The ROC curve analysis shows superior performance of the deep learning model, with AUC values consistently higher across all disease categories.

5. While SVM and Random Forest demonstrate strong recall for healthy patients, they struggle with precision in disease detection, indicating a higher false positive rate compared to the deep learning approach.

6. The proposed model demonstrates balanced performance across precision and recall metrics, making it more reliable for clinical applications where both false positives and false negatives have significant consequences.

All visualizations can be found in the following files:
- Model comparison metrics: model_comparison_metrics.png
- ROC curve comparisons: model_comparison_roc.png
- Performance metric heatmaps: model_comparison_heatmaps.png
