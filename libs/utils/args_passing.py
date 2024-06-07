import toml

from pathlib import Path

def args_control(path:Path):
    
    args = toml.load(path)
     
    return args