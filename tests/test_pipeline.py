"""Regression checks on the real Kaggle files."""
import numpy as np
import pytest
import torch

from src.data import read_data, split_data
from src.model import HousePriceMLP
from src.preprocess import load_preprocessor, prepare_data, save_preprocessor, transform_test
from src.predict import validate_submission
from src.evaluate import predict_loader
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


@pytest.fixture(scope="module")
def prepared():
    train, test = read_data()
    return prepare_data(split_data(train, test, seed=42))


def test_model_output_shape(prepared):
    model = HousePriceMLP(prepared.x_train.shape[1]).eval()
    with torch.no_grad():
        assert model(torch.from_numpy(prepared.x_val[:5])).shape == (5, 1)
        assert model(torch.from_numpy(prepared.x_val[:1])).shape == (1, 1)


def test_preprocessing_is_finite_and_excludes_id_target(prepared):
    assert "Id" not in prepared.feature_columns
    assert "SalePrice" not in prepared.feature_columns
    assert prepared.y_train.shape == (1168, 1)
    for x in (prepared.x_train, prepared.x_val, prepared.x_test):
        assert x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all()


def test_validation_and_test_changes_do_not_change_fit_statistics():
    train, test = read_data()
    split = split_data(train, test, seed=42)
    original = prepare_data(split)
    split.x_val = split.x_val.copy()
    split.x_val.loc[:, "LotArea"] = 1e12
    split.x_val.loc[:, "Neighborhood"] = "UNSEEN_CATEGORY"
    split.x_test = split.x_test.copy()
    split.x_test.loc[:, "LotArea"] = 1e12
    split.x_test.loc[:, "Neighborhood"] = "UNSEEN_CATEGORY"
    modified = prepare_data(split)
    np.testing.assert_allclose(original.x_train, modified.x_train)
    assert not np.array_equal(original.x_val, modified.x_val)
    assert not np.array_equal(original.x_test, modified.x_test)
    numeric = original.preprocessor.named_transformers_["num"].named_steps["scaler"]
    numeric_changed = modified.preprocessor.named_transformers_["num"].named_steps["scaler"]
    np.testing.assert_allclose(numeric.mean_, numeric_changed.mean_)
    assert np.isfinite(modified.x_val).all()
    assert np.isfinite(modified.x_test).all()


def test_checkpoint_roundtrip_preserves_predictions(prepared, tmp_path):
    model = HousePriceMLP(prepared.x_train.shape[1]).eval()
    x = torch.from_numpy(prepared.x_val[:7])
    with torch.no_grad():
        expected = model(x).clone()
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": prepared.x_train.shape[1],
                "hidden_dims": [128, 64, 32], "dropout": [0.15, 0.10]}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = HousePriceMLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"]).eval()
    restored.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected)


def test_preprocessor_roundtrip_and_prediction_shape(prepared, tmp_path):
    path = save_preprocessor(prepared, tmp_path / "preprocessor.joblib")
    _, test = read_data()
    reloaded = transform_test(test, load_preprocessor(path))
    np.testing.assert_allclose(reloaded, prepared.x_test)
    loader = DataLoader(TensorDataset(torch.from_numpy(reloaded)), batch_size=32, shuffle=False)
    model = HousePriceMLP(reloaded.shape[1])
    assert predict_loader(model, loader, torch.device("cpu")).shape == (len(test),)


def test_submission_validation_rejects_bad_values():
    ids = pd.Series([1461, 1462])
    good = pd.DataFrame({"Id": ids, "SalePrice": [100000.0, 200000.0]})
    validate_submission(good, ids)
    bad_cases = [
        good[["SalePrice", "Id"]],
        good.iloc[:1],
        good.assign(Id=[1462, 1461]),
        good.assign(SalePrice=[np.nan, 1]),
        good.assign(SalePrice=[np.inf, 1]),
        good.assign(SalePrice=[-1, 1]),
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            validate_submission(bad, ids)
