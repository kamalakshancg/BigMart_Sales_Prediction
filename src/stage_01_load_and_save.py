from src.utils.all_utils import read_yaml,create_dir
import os
import pandas as pd
import argparse

def load_save(config_path):
    #path_to_config = "/home/kamli/DVC_Pipline/BigMartDVC/config/config.yaml"

    contents = read_yaml(config_path)
    
    data = pd.read_csv(contents['data_source'],sep=";")

    artifacts_dir = contents['artifacts']['artifacts_dir']
    local_dir = contents['artifacts']['raw_local_dir']
    raw_local_file= contents['artifacts']['raw_local_file']

    raw_local_dir_path = os.path.join(artifacts_dir,local_dir)

    raw_local_file_path= os.path.join(raw_local_dir_path,raw_local_file)

    create_dir(dirs=[raw_local_dir_path])
    
    
    data.to_csv(raw_local_file_path,sep=',',index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    load_save(config_path=parsed_args.config)

   