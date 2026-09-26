# Machine-learning meta-analysis: locked analysis report

## Analysis population

The primary analysis used 163 comparable effect sizes from 34 reports and 32 independent cohort/sample clusters. The proxy-augmented sensitivity analysis used 254 rows from 48 reports and 45 clusters; 91 rows were explicitly flagged as proxy statistics and were not mixed into the primary evidence set.

## Validation and model selection

All rows from the same cohort/sample cluster were kept in the same fold. Four regression algorithms (ridge, elastic net, random forest and histogram gradient boosting) were compared with a training-fold mean comparator using 5-fold outer validation repeated 5 times. Hyperparameters were selected exclusively in 4-fold grouped inner validation. Training and evaluation were group-balanced so that each independent cluster contributed equal total weight. No effect size, confidence interval, standard error, sampling variance or record identifier was supplied as a predictor.

## Primary result

The lowest-RMSE non-null algorithm was Ridge. Its mean group-balanced RMSE was 0.191 (repeat range approximated by the 2.5th–97.5th percentiles, 0.182 to 0.204), compared with 0.188 for the null comparator. Mean MAE was 0.135, mean predictive Q² was -0.084, and the mean calibration slope was 0.35. No candidate model improved on the training-mean comparator under cohort-grouped validation; the ML findings should therefore be treated as exploratory heterogeneity mapping rather than a validated prediction tool.

The final full-data refit used: `{"model__alpha": 100.0}`. This refit is provided for interpretation and future external validation; its apparent fit is not used as evidence of performance.

## Exploratory predictor dependence

Cross-validated group-block permutation ranked the leading predictors as Adjusted (ΔRMSE=0.009), EF_Measure_Type (ΔRMSE=0.005), Log_N (ΔRMSE=0.004), Study_design (ΔRMSE=0.002), Year (ΔRMSE=0.001). These values quantify loss of held-out predictive accuracy after disrupting a predictor while preserving clustered validation. They are not causal effects and should not be interpreted as intervention targets. Partial-dependence plots are likewise descriptive full-data refits.

## Recommended manuscript interpretation

The ML analysis estimates whether study, exposure and outcome characteristics generalize to entirely unseen cohorts. It does not classify individual children and does not establish causal moderators. Proxy-statistic results are sensitivity evidence only because regression coefficients and other non-correlation estimates are not guaranteed to share a common scale with Fisher-transformed correlations.
