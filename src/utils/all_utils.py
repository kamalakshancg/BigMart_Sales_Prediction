import yaml
import os


def read_yaml(path_to_yaml):
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def create_dir(dirs):
    for dires in dirs:
        os.makedirs(dires)
        print(dirs,"directory created")
