"""
The Config class is responsible for creating a nested configuration object by 
merging a default configuration file and an optional user-defined configuration 
file, both in YAML format. The class loads both configuration files and merges 
them, with the user-defined configuration file taking precedence over the 
default configuration file. 

The resulting configuration object is created recursively, where nested 
dictionaries are converted to nested Config objects. The class provides an 
interface to access the configuration options as object attributes.

Suppose we have the following YAML file, config.yaml:

--yaml--
general:
  skip_error: !!bool false

path:
  output_dir: !!str out_dir
----

--python--

config = Config('config.yaml')
skip_error = config.general.skip_error
output_dir = config.path.output_dir
----

"""

from typing import Dict
import argparse
import yaml

def default_options():
    """
    Parse command-line arguments for default and user configuration file paths.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--default-config-file", '-b', 
        type=str,
        default='configs/config_default.yaml',
        help="Path to the base configuration file. Defaults to 'config.yaml'.")
    parser.add_argument(
        "--user-config-file", '-u', 
        type=str,
        help="Path to the user-defined configuration file.")
    args = parser.parse_args()
    return args


def merge_config_dicts(d1: Dict, d2: Dict):
    """
    Merge two configuration dictionaries, with the second dictionary's values
    taking precedence over the first one's values.

    Args:
        d1 (dict): The first configuration dictionary.
        d2 (dict): The second configuration dictionary.

    Returns:
        dict: The merged configuration dictionary.
    """
    result = d1.copy()
    for key, value in d2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config_dicts(result[key], value)
        else:
            result[key] = value
    return result


class Config():
    """Convert a nested config dictionary to a nested config object."""

    def __init__(self, default_config_file: str = 'config.yaml',
                       user_config_file: str = None):
        if not default_config_file:
            return
        print(f"[INFO] Loading config files:")
        print(f"[INFO]    Default config: {default_config_file}")
        print(f"[INFO]    User config: {user_config_file}")

        with open(default_config_file) as f:
            default_config = yaml.safe_load(f)
        if user_config_file:
            with open(user_config_file) as f:
                user_config = yaml.safe_load(f)
            config = merge_config_dicts(default_config, user_config)
        else:
            config = default_config

        self.__dict__.update(self._nested_dicts_to_objects(config))

    def _nested_dicts_to_objects(self, config_dict: Dict) -> Dict:
        """
        Convert a nested dictionary to a nested config object.

        Args:
            config_dict (Dict): A nested dictionary.

        Returns:
            Dict: A nested config object.

        """
        result = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                obj = Config(None, None)
                obj.__dict__.update(self._nested_dicts_to_objects(value))
                result[key] = obj
            else:
                result[key] = value
        return result