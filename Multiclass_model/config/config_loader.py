import yaml
import os

CONFIG_PATH = "./Multiclass_model/config"

def load_config(config_name = "config.yaml"):
    with open(os.path.join(CONFIG_PATH,config_name)) as file:
        config = yaml.safe_load(file)
        
    return config

        
    
        
    