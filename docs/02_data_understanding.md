# Data Understanding

The real Kaggle files contain 1,460 training rows × 81 columns and 1,459 test rows × 80 columns. After removing `Id` and `SalePrice`, there are 79 properties: 36 numeric and 43 categorical. Both files have zero duplicate rows. `Id` is unique in each file.

`SalePrice` has mean 180,921.20, median 163,000, standard deviation 79,442.50 and skewness 1.883. `log1p(SalePrice)` skewness is 0.121. The target histograms in `outputs/figures/target_histograms.png` show the effect.

The largest missing counts in training are `PoolQC` (1,453), `MiscFeature` (1,406), `Alley` (1,369), `Fence` (1,179), `FireplaceQu` (690), and numeric `LotFrontage` (259). Several `NA` values mean an amenity is absent in the data dictionary, but the planned most-frequent imputer treats them as missing; that loses potentially useful information. The CSV loader preserves the literal `None` category for `MasVnrType` by treating only empty strings and `NA` as missing.

The strongest numeric correlations with `SalePrice` are `OverallQual` (0.791), `GrLivArea` (0.709), `GarageArea` (0.623), `TotalBsmtSF` (0.614), and `YearBuilt` (0.523). The EDA includes scatter plots for living area, basement area, garage area and build year; boxplots for overall quality and neighborhood. `GrLivArea` above 4,000 square feet with price below 300,000 flags Id 524 and 1299 as notable outlier candidates. Both are retained in the baseline; removal would need a documented, validated reason.

Reproduce plots and printed statistics with `notebooks/01_eda.ipynb`, which calls `src/eda.py`.
