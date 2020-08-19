import configparser
import yaml


def get_ini_configs(settings="default.ini"):
    """Input settings in ini-format and return python configuration parser object."""
    config = configparser.ConfigParser()
    config.read(f"./shorelineforecasting/configurations/{settings}")
    return config


def get_yaml_configs(settings="default.yml"):
    """"Input configurations in YAML-format and return python dict-style configs."""
    with open(f"./shorelineforecasting/configurations/{settings}") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data



if __name__ == "__main__":
    configs = get_ini_configs()
    print(configs.sections())

    yummy = get_yaml_configs()
    print(yummy.items())
    print(type(yummy['run']['logger']))
