import os
import sys


module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

configuration_dir = os.path.join(module_path, 'shorelineforecasting/configurations')
data_dir = "/media/storage/data/shorelines"
report_dir = "/media/storage/Documents/papers/shoreline-forecasting"
model_dir = os.path.join('/media/storage/data/shorelines/models')


if __name__ == "__main__":
    print(module_path)
