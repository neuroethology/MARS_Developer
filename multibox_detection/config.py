import yaml
from easydict import EasyDict as easydict
 
# TODO: Add in default values 
 
def parse_config_file(path_to_config):
  
  with open(path_to_config) as f:
    
    cfg = yaml.safe_load(f)
  
  return easydict(cfg)  