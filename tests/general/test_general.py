import pytest
import os


def test_correct_env_path():
    root_dir = os.getenv('ROOTDIR')
    assert root_dir is not None


def test_correct_project_struct():
    root_dir = os.getenv('ROOTDIR')
    logs_folder = os.path.join(root_dir, 'logs')
    assert os.path.exists(logs_folder)
    reports_folder = os.path.join(root_dir, 'reports')
    assert os.path.exists(reports_folder)
    last_trained = os.path.join(root_dir, 'last_trained')
    assert os.path.exists(last_trained)
    predictions = os.path.join(root_dir, 'predictions')
    assert os.path.exists(predictions)

