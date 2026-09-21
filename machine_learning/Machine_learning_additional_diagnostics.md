# Additional machine-learning diagnostics

## Scope

Five figures were added to extend the three original machine-learning figures without changing the locked primary outcome, cohort-grouped validation design or selected model. The complete set now contains eight figures. ML1, ML2, ML4 and ML6 are recommended for the main text; ML3, ML5, ML7 and ML8 are recommended for Supplementary Information.

## Main findings

- The learning curve improved as independent training support increased. Mean held-out ridge RMSE decreased from 0.200 with 8 training clusters to 0.185 with 27 clusters. At 27 clusters, mean RMSE was 0.013 lower than the training-mean comparator, but its empirical 95% interval for the paired difference (−0.043 to 0.011) still included zero. Mean predictive Q² remained negative (−0.191), indicating that the evidence base did not yet support stable prediction in unseen cohorts.
- Prediction error was concentrated in several independent samples. The largest cluster-level MAEs were observed for Xu (2023; 0.567), PSKC (0.328) and Luo (2021; 0.308). These are transportability diagnostics, not study-quality judgements.
- Among category combinations supported by at least three independent clusters, the largest group-balanced MAEs occurred for global executive function with interactive/gaming exposure (0.323; g=3, k=3), inhibitory control with interactive/gaming exposure (0.243; g=4, k=11), and inhibitory control with social media/multitasking exposure (0.177; g=4, k=17). Sparse combinations were explicitly flagged rather than interpreted.
- Model selection was unstable across algorithms. Ridge won 9 of 25 inner-validation contests, elastic net 8 and random forest 7. Ridge improved on the comparator in 18 of 25 outer folds, but the mean RMSE advantage was only 0.0024. A strong ridge penalty (α=100) was selected in 24 of 25 folds, consistent with substantial shrinkage and weak multivariable signal.
- In the descriptive full-data ridge refit, adjusted-status, log sample size and EF measurement type produced the largest mean absolute linear contributions. These full-data contributions are not held-out importance estimates; the permutation intervals in Figure ML2 remain the primary explanation analysis and crossed zero for all predictor blocks.

## Interpretation for the manuscript

The additional analyses do not rescue a poorly generalising model. Instead, they identify why performance was limited and where new evidence would add most value. The gradual learning-curve improvement suggests that prediction may benefit from a larger and more diverse set of independent cohorts, whereas the error-support matrix shows that several clinically meaningful exposure–outcome combinations remain supported by only three to four clusters. The appropriate conclusion is therefore that current study-level moderators are insufficient for reliable cross-cohort prediction, not that screen exposure has no association with executive function.

ROC curves were not produced because the outcome is continuous Fisher's z and no clinically justified binary target was prespecified. PCA cluster plots were also omitted because they would visualise predictor-space structure rather than validate prediction of the meta-analytic outcome.
