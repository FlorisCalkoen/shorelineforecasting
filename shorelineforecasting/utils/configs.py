import configparser

def get_configs(settings = "default.ini" ):
    """"""
    config = configparser.ConfigParser()
    config.read(f"./shorelineforecasting/configurations/{settings}")
    return config

if __name__ == "__main__":
    configs = get_configs()
    print(configs.sections())