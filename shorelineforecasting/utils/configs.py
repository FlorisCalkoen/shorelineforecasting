import yaml
import configparser


def get_ini_configs(settings="default.ini"):
    """Input settings in ini-format and return python configuration parser object."""
    config = configparser.ConfigParser()
    config.read(f"./configurations/{settings}")
    return config


def get_yaml_configs(settings="default.yml"):
    """"Input configurations in YAML-format and return python dict-style configs."""
    with open(f"./configurations/{settings}") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
