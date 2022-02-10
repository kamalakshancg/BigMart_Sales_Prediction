from operator import index
import pandas as pd
import os
import yaml
import argparse
from src.utils.all_utils import read_yaml,create_dir
from sklearn.model_selection import train_test_split
import logging

def Split(config_path):
    logger = logging.getLogger('')
    f_handler = logging.FileHandler('Split_data.log')
    f_handler.setLevel(logging.ERROR)
    f_format = logging.Formatter('%(asctime)s %(levelname)s [%(funcName)s] %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    contents = read_yaml(config_path)

    artifacts_dir = contents['artifacts']['artifacts_dir']
    raw_local_split_dir = contents['artifacts']['raw_local_split_dir']
    raw_local_dir = contents['artifacts']['raw_local_dir']
    #raw_local_data_file = contents['artifacts']['raw_local_file']
    raw_local_data_file = contents['artifacts']['data_file']
    
    dir_path = os.path.join(artifacts_dir,raw_local_dir)
    raw_local_data_file_path = os.path.join(dir_path,raw_local_data_file)

    test_file_name = contents['artifacts']['local_test_file']
    train_file_name = contents['artifacts']['local_train_file']

   

    raw_local_split_dir_path = os.path.join(artifacts_dir,raw_local_split_dir)
    test_file_path = os.path.join(raw_local_split_dir_path,test_file_name)
    train_file_path = os.path.join(raw_local_split_dir_path,train_file_name)
    logger.debug("test_file_path:",test_file_path," ","train_file_path: ",train_file_path)
    
    try:
        create_dir(dirs=[raw_local_split_dir_path])
    except:
        logger.error("Error while creating folder or crating path")

    data = pd.read_csv(raw_local_data_file_path)

    train,test = train_test_split(data,test_size=0.2,random_state=42)
    logger.error("data splited succesfully")
    train.to_csv(train_file_path,index=False)
    test.to_csv(test_file_path,index=False)
    logger.error("train and test data saved as csv file")

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    Split(config_path=parsed_args.config)
