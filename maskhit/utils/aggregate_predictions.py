# Example usage:
# python utils/aggregate_predictions.py --study_name "ibd_project" --timestr "2023_5_30_new-test-" --user-config-file "configs/config_ibd_train.yml"
# If either study_name or timestr is not provided, the script will exit with an error code.

import argparse
from maskhit.trainer.metrics import aggregate_predictions
from config import Config

def main(study_name, timestr, config_file):
    """
    Runs the aggregate_predictions function with given study_name and timestr.

    Parameters:
    study_name (str): The name of the study.
    timestr (str): The timestamp string.

    If either study_name or timestr is None, the function returns an error.
    """

    if study_name is None or timestr is None:
        print("Error: 'study_name' and 'timestr' arguments are required.")
        exit(1)
    classes = config.dataset.classes.split(',')
    aggregate_predictions(study_name, timestr, classes = classes)
    exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run aggregate_predictions with study_name and timestr.")
    parser.add_argument("--study_name", type=str, help="Name of the study")
    parser.add_argument("--timestr", type=str, help="Timestamp string")
    parser.add_argument(
        "--default-config-file", 
        type=str,
        default='configs/config_default.yaml',
        help="Path to the base configuration file. Defaults to 'config.yaml'.")
    parser.add_argument(
        "--user-config-file", 
        type=str,
        help="Path to the user-defined configuration file.")

    args = parser.parse_args()
    config = Config(args.default_config_file, args.user_config_file)

    main(args.study_name, args.timestr, config)
