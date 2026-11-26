# Scientific Evaluation Report: rice_leaf (mobilenet_v3_small)
**Evaluation Date:** 2026-10-24T11:12:37.988574+00:00  
**Test Set Samples:** 16  

## Summary Performance
- **Overall Accuracy:** 43.75%
- **Macro Precision:** 28.33%
- **Macro Recall:** 40.00%
- **Macro F1-Score:** 0.3241
- **Weighted F1-Score:** 0.3507
- **Inference Latency (Avg):** 6.8 ms / sample

## Per-Class Breakdown
| Condition | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `bacterial_leaf_blight` | 60.0% | 100.0% | 0.7500 | 6.0 |
| `brown_spot` | 0.0% | 0.0% | 0.0000 | 5.0 |
| `leaf_smut` | 25.0% | 20.0% | 0.2222 | 5.0 |

## Confusion Matrix
Rows represent ground-truth classes; columns represent model predictions: ['bacterial_leaf_blight', 'brown_spot', 'leaf_smut']

```text
     6     0     0
     2     0     3
     2     2     1
```