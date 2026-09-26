# Data Preparation

`src.data.split_data` removes `Id` and `SalePrice`, computes `log1p(SalePrice)`, then makes a fixed 80/20 split with seed 42. The test `Id` series is retained in source order.

`src.preprocess.prepare_data` fits a `ColumnTransformer` **only on the training fold**. Numeric columns use median imputation then `StandardScaler`; categorical columns use most-frequent imputation then `OneHotEncoder(handle_unknown="ignore")`. Validation and test are transformed by this fitted object. The fitted transformer and source feature column order are serialized together for inference.

Dense one-hot output is appropriate for this 1,460-row dataset and avoids sparse-to-tensor conversion in the baseline. The actual transformed shapes are train `(1168, 286)`, validation `(292, 286)`, and test `(1459, 286)`; their dense arrays use about 1.27, 0.32, and 1.59 MiB. Arrays and tensors are `float32`; target tensors have `(N, 1)` shape. The train loader shuffles with seed 42, while validation and test loaders do not. If a final train batch contains one sample, it is dropped for BatchNorm; evaluation still uses every training sample.

This baseline deliberately treats `NA` as missing, though the data dictionary sometimes means an absent amenity. It does not add engineered features or remove outliers. Notebook `02_preprocessing.ipynb` prints actual dimensions and dense memory use.
