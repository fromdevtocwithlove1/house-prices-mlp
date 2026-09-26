# Business Understanding

The Kaggle House Prices task is supervised regression: estimate a continuous `SalePrice` from the home's recorded properties. The competition requires one `SalePrice` for each test `Id`. `Id` is an identifier, not a model feature.

The model predicts `log1p(SalePrice)` so large prices have less leverage during squared-error training; predictions return to price units with `expm1`. The main local metric is RMSE on log-price, and the winning configuration is the one with the lowest validation RMSE on a fixed 80/20 holdout. Kaggle's hidden test score is external and must not guide the local split or preprocessing.

Deliverable: `outputs/submission.csv` with exactly `Id,SalePrice`. This project keeps a reusable pipeline and records each CRISP-DM phase in `docs/`.
