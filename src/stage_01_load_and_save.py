from src.utils.all_utils import read_yaml,create_dir
import os
import pandas as pd
import argparse
import logging

def load_save(config_path):
    logger = logging.getLogger('')
    f_handler = logging.FileHandler('load_save.log')
    f_handler.setLevel(logging.ERROR)
    f_format = logging.Formatter('%(asctime)s %(levelname)s [%(funcName)s] %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)


    contents = read_yaml(config_path)
    
    data = pd.read_csv(contents['data_source'],sep=";")

    artifacts_dir = contents['artifacts']['artifacts_dir']
    local_dir = contents['artifacts']['raw_local_dir']
    raw_local_file= contents['artifacts']['raw_local_file']

    
    try:
        raw_local_dir_path = os.path.join(artifacts_dir,local_dir)
        raw_local_file_path= os.path.join(raw_local_dir_path,raw_local_file)
        create_dir(dirs=[raw_local_dir_path])
        logger.error("Folder created")
    except:
        logger.error("Error while creating folder or crating path")
    
    
    data.to_csv(raw_local_file_path,sep=',',index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    load_save(config_path=parsed_args.config)

   