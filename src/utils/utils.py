import os


def set_env_variable():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    root_project_dir = os.path.dirname(os.path.dirname(utils_dir))
    os.environ["ROOT_PROJECT_DIR"] = root_project_dir


def prepare_work_env():
    """
    func for setting workspace
    :return:
    """
    set_env_variable()
    root_project_dir = os.environ["ROOT_PROJECT_DIR"]
    os.environ["PATH"] += root_project_dir
    print("Creating logs folder...")
    logs_dir = os.path.join(root_project_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
        print("Done")
    else:
        print("logs folder already exists!")
    print("Creating reports folder...")
    reports_dir = os.path.join(root_project_dir, 'reports')
    if not os.path.exists(reports_dir):
        os.mkdir(reports_dir)
        print("Done")
    else:
        print("reports folder already exists!")
    print("Creating latest trained model folder...")
    reports_dir = os.path.join(root_project_dir, 'latest_trained')
    if not os.path.exists(reports_dir):
        os.mkdir(reports_dir)
        print("Done")
    else:
        print("latest trained model folder already exists!")
    print("Creating result predictions folder...")
    preds_dir = os.path.join(root_project_dir, 'predictions')
    if not os.path.exists(preds_dir):
        os.mkdir(preds_dir)
        print("Done")
    else:
        print("result predictions folder already exists!")


if __name__ == "__main__":
    prepare_work_env()
