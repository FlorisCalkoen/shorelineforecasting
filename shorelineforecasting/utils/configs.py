import os
import yaml
import configparser
import pandas as pd


class GluonConfigs():
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        return pd.read_csv("/media/storage/data/shorelines/time-series-gluonts-prepared.csv")


def get_predictor_id(filename="shorelineforecasting/varstore.dat", just_read=False):
    root_dir = os.path.abspath(os.path.join(os.pardir, os.pardir))
    fpath = os.path.join(root_dir, filename)
    with open(fpath, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0)
        if just_read is False:
            val += 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

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
