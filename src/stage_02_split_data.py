import pandas as pd
import os
import yaml
import argparse
from src.utils.all_utils import read_yaml,create_dir
from sklearn.model_selection import train_test_split




def Split(config_path):

    contents = read_yaml(config_path)

    artifacts_dir = contents['artifacts']['artifacts_dir']
    raw_local_split_dir = contents['artifacts']['raw_local_split_dir']
    raw_local_dir = contents['artifacts']['raw_local_dir']
    raw_local_data_file = contents['artifacts']['raw_local_file']
    
    dir_path = os.path.join(artifacts_dir,raw_local_dir)
    raw_local_data_file_path = os.path.join(dir_path,raw_local_data_file)

    test_file_name = contents['artifacts']['local_test_file']
    train_file_name = contents['artifacts']['local_train_file']

   

    raw_local_split_dir_path = os.path.join(artifacts_dir,raw_local_split_dir)
    test_file_path = os.path.join(raw_local_split_dir_path,test_file_name)
    train_file_path = os.path.join(raw_local_split_dir_path,train_file_name)

    create_dir(dirs=[raw_local_split_dir_path])

    data = pd.read_csv(raw_local_data_file_path)

    train,test = train_test_split(data,test_size=0.2,random_state=42)
    
    train.to_csv(train_file_path)
    test.to_csv(test_file_path)

    print("data splited succesfully")
    





    

    



if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    Split(config_path=parsed_args.config)
