import toml
import shutil

from pathlib import Path

def args_control(
    args_file:Path,
    save_dir: Path
):
    
    args = toml.load(args_file)

    if save_dir is not None:

        file_name = save_dir / args_file.name
        shutil.copyfile(args_file, file_name)

    return args