import pandas as pd
import pytest
from hydra.experimental import initialize, compose
from src.train import train
from src.predict import predict
from src.data import get_target_data
from src.config import Config
import os
import pickle
from pathlib import Path

PROJECT_ROOT = os.getenv('ROOTDIR')
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path('.')
PROJECT_ROOT = Path(PROJECT_ROOT)


@pytest.fixture
def conf(tmpdir) -> Config:
    conf_path = '../../conf'
    with initialize(config_path=str(conf_path)):
        config_params: Config = compose(config_name='common_conf')
    reports_path = tmpdir / config_params.general.reports_path
    tmpdir.mkdir(config_params.general.reports_path)
    models_path = tmpdir / config_params.general.models_path
    tmpdir.mkdir(config_params.general.models_path)
    config_params.general.reports_path = str(reports_path)
    config_params.general.models_path = str(models_path)
    tmpdir.mkdir('predictions')
    return config_params


@pytest.fixture
def dataset(conf: Config) -> pd.DataFrame:
    data_path = conf.data.dataset_path
    dataset_filename = conf.data.dataset_fname
    dataset_path = Path(data_path) / dataset_filename
    X_test, _ = get_target_data(pd.read_csv(dataset_path), 'condition')
    return X_test


@pytest.fixture
def trained_model(conf: Config):
    train(conf)
    model_path = Path(conf.general.models_path) / 'last_trained_model.pickle'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def test_models(trained_model, dataset, conf):
    assert (Path(conf.general.models_path) / 'last_trained_model.pickle').exists()
    assert Path(conf.general.reports_path).exists()
    preds = predict(trained_model, dataset)
    assert isinstance(preds, pd.Series)
