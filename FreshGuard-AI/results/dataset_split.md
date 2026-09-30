# Dataset Split Summary

### Scope Decision Justification
For this initial deployment, we have scoped the dataset to the three most commercially dominant fruits (Apple, Banana, Orange). This ensures our flagship Attention-CNN model can focus on learning highly robust, generalized freshness features without being penalized by the high intra-class variance of the full 12-class FruitsGB dataset. Furthermore, relabeling the 'Good/Bad' dataset nomenclature to 'Fresh/Rotten' specifically aligns the model's vocabulary with our problem statement—focusing on food safety and spoilage detection rather than generic cosmetic defects.

### Image Counts by Split

| Class | Train (70%) | Val (15%) | Test (15%) | Total |
|---|---|---|---|---|
| Apple_Fresh | 700 | 150 | 150 | 1000 |
| Apple_Rotten | 700 | 150 | 150 | 1000 |
| Banana_Fresh | 700 | 150 | 150 | 1000 |
| Banana_Rotten | 700 | 150 | 150 | 1000 |
| Orange_Fresh | 700 | 150 | 150 | 1000 |
| Orange_Rotten | 700 | 150 | 150 | 1000 |
| **TOTAL** | **4200** | **900** | **900** | **6000** |
