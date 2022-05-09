import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import requests

DEFAULT_PATH = os.path.join(os.getenv('ROOTDIR'), '/data/heart_cleveland_upload.csv')


def download_from_kaggle(link, path):
    response = requests.get(link)
    with open(path, 'w') as f:
        f.write(response.text)


def callback_run(arguments):
    download_from_kaggle(arguments.link, arguments.path_to_load)


def setup_parser(parser):
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS,
                        help='Show help info')
    parser.add_argument('-l', '--link',
                        help='link for download data',
                        required=True)
    parser.add_argument('-p', '--path_to_load',
                        help='path to data',
                        default=DEFAULT_PATH)
    parser.set_defaults(callback=callback_run)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Download data",
        description="Script for downloading data",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)
