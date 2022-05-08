import logging.config
import os
import pickle
from datetime import datetime
import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

from src.config import Config
from utils.utils import set_env_variable
from data import get_splitted_data, get_target_data

from predict import predict


ROOT_PROJECT_DIR = os.getenv('ROOTDIR')
logging.config.fileConfig(f'{ROOT_PROJECT_DIR}/conf/logging.conf')
logger = logging.getLogger('heart_dis_classifier')


def train_model(model: BaseEstimator, X_train: pd.DataFrame,
                y_train: pd.Series) -> BaseEstimator:
    """
    Training model function
    :param model: chosen for training model
    :param X_train: train data
    :param y_train: train target
    :return:
    prepared model
    """
    logger.info('Model training started')
    model.fit(X_train, y_train)
    logger.info('Model training finished')
    return model


def validate(model: BaseEstimator, X_val: pd.DataFrame,
             y_val: pd.Series, score_func: DictConfig) -> float:
    """
    Validation model func
    :param model: chosen for validation model
    :param X_val: validation data
    :param y_val: validation target
    :param score_func: score func
    :return:
    score
    """
    logger.info('Model validation started')
    preds = model.predict(X_val)
    logger.info('Model validation finished')
    return hydra.utils.call(score_func, preds, y_val)


def create_report(model: BaseEstimator,  config_params: Config, project_root: str, score: dict):
    out_dir = os.path.join(project_root, config_params.general.reports_path)
    logger.info(f'Report creation started')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    report_folder = f'report_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    report_path = os.path.join(out_dir, report_folder)
    last_model_path = os.path.join(project_root, 'latest_trained')
    os.mkdir(report_path)
    config_file = os.path.join(report_path, 'general_config.yaml')
    model_dump = os.path.join(report_path, 'model.pickle')
    last_model_dump = os.path.join(last_model_path, 'last_trained_model.pickle')
    score_report_path = os.path.join(report_path, 'score_report.yaml')
    with open(score_report_path, 'w') as f:
        f.write(yaml.dump(score))
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg=config_params))
    with open(model_dump, 'wb') as f:
        pickle.dump(model, f)
    if os.path.exists(last_model_dump):
        os.remove(last_model_dump)
        logger.info(f'previous trained model removed')
    with open(last_model_dump, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'========== Finished model train ==========')
    logger.info(f'Report stored in {report_path}')


@hydra.main(config_path='../conf', config_name='common_conf')
def train(config_params: Config):
    set_env_variable()
    project_root = os.environ["ROOT_PROJECT_DIR"]
    logger.info(f'========== Start model train ==========')
    logger.info("Detailed info:")
    logger.info(f'---------------------------------------')
    logger.info("Model:")
    for param in config_params.model.keys():
        logger.info(f"{param}: {config_params.model[param]}")
    logger.info(f'---------------------------------------')
    logger.info("Data:")
    full_path = os.path.join(project_root, config_params.data['dataset_path'])
    logger.info(f"{'dataset path'}: {full_path}")
    full_path = os.path.join(full_path, config_params.data['dataset_fname'])
    logger.info(f'---------------------------------------')
    logger.info("Data split params:")
    for param in config_params.data_work.keys():
        logger.info(f"{param}: {config_params.data_work[param]}")
    logger.info(f'---------------------------------------')
    logger.info("Preparing data for train:")
    df = pd.read_csv(full_path)
    train_df, val_df, test_df = get_splitted_data(df, config_params.data_work)
    model = hydra.utils.instantiate(config_params.model)
    X_train, y_train = get_target_data(train_df, config_params.data_work.target_name)
    logger.info("Data prepared")
    model = train_model(model, X_train, y_train)
    score_func = config_params.general.score_func
    X_val, y_val = get_target_data(train_df, config_params.data_work.target_name)
    val_score = validate(model, X_val, y_val, score_func)
    logger.info(f"Model score: {val_score}")
    X_test, y_test = get_target_data(test_df, config_params.data_work.target_name)
    preds = predict(model, X_test)
    score_res = classification_report(y_true=y_test, y_pred=preds, output_dict=True)
    create_report(model,  config_params, project_root, score_res)


if __name__ == '__main__':
    train()