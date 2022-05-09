import os
import pickle

import logging.config
import sys


import pandas as pd
from sklearn.base import BaseEstimator
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from src.data import get_target_data

ROOT_PROJECT_DIR = os.getenv('ROOTDIR')
logging.config.fileConfig(f'{ROOT_PROJECT_DIR}/conf/logging.conf')
logger = logging.getLogger('heart_dis_classifier')
DEFAULT_MODEL_PATH = 'last_trained/last_trained_model.pickle'
DEFAULT_RES_PATH = 'predictions'


def predict(model: BaseEstimator, X_test: pd.DataFrame) -> pd.Series:
    """
    function for predict
    :param model: chosen for predict model
    :param X_test:data for predict
    :return:
    predicted data
    """
    logger.info("Model prediction started")
    preds = model.predict(X_test)
    preds = pd.Series(preds, index=X_test.index, name='preds')
    logger.info("Model prediction finished")
    return preds


def callback_run(arguments):
    predict_pipeline(arguments.model_path, arguments.data_path, arguments.target, arguments.res_path)


def setup_parser(parser):
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
                        help='Show help info')
    parser.add_argument('-mp', '--model_path',
                        help='Path to trained model for prediction',
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument('-dp', '--data_path',
                        help='Path to data file for prediction',
                        required=True)
    parser.add_argument('-r', '--res_path',
                        help='Result path for prediction. If this key is not set, predictions will be saved in '
                             'default folder',
                        required=False,
                        default=DEFAULT_RES_PATH)
    parser.add_argument('-t', '--target',
                        help='Target column in data for prediction. Use this key '
                             'if you want send data with target column',
                        required=False,
                        default='condition')
    parser.set_defaults(callback=callback_run)


def predict_pipeline(model_path: str, data_path: str, target: str, res_path: str):
    """
    Main function
    :return:
    """
    project_root = os.environ["ROOTDIR"]
    if model_path == DEFAULT_MODEL_PATH:
        logger.info("Used last trained model")
        model_path = os.path.join(project_root, model_path)
    if not os.path.isfile(model_path):
        logger.error("Wrong path to model specified")
        sys.exit(1)
    if res_path == DEFAULT_RES_PATH:
        res_path = os.path.join(project_root, res_path)
        logger.info(f"result will be saved in {res_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except pickle.UnpicklingError as e:
        logger.error(f'an error occurred while opening file {model_path}: {str(e)}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'an error occurred while opening file {model_path}: {str(e)}')
        sys.exit(1)
    if not os.path.isfile(data_path):
        logger.error("Wrong path to data specified")
        sys.exit(1)
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f'an error occurred while opening file {data_path}: {str(e)}')
        sys.exit(1)
    X_test, y_test = get_target_data(df, target)
    preds = predict(model, X_test)
    if not os.path.isfile(res_path):
        res_fname = f'result_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.csv'
        res_path = os.path.join(res_path, res_fname)
    preds.to_csv(res_path)
    logger.info(f'result saved')


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Ml project for determine heart disease",
        description="Homework 1. Ml project for determine heart disease",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


