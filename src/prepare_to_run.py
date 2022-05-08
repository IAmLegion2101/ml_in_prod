import os

def prepare_work_env():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(os.path.dirname(utils_dir), 'logs')
    os.mkdir(logs_dir)