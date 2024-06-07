
import torch

import numpy as np


from pathlib import Path

from libs.architecture import WaveNet
from libs.utils import args_control
from libs.data import fast_dataloader
from libs.train import training
from libs.train.trainer import Validation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_args = args_control(Path("./arguments.toml"), Path("../../Outputs"))

SIZE_1, SIZE_2, SIZE_3 = 2*project_args["batch_size"], 2, 4096
pseudo_data = torch.Tensor(np.random.normal(0, 1, (SIZE_1, SIZE_2, SIZE_3)))
pseudo_target = torch.Tensor(np.random.choice([0, 1], size=(SIZE_1, 1)))


def main(
    n_ifos=project_args["n_ifos"], 
    batch_size=project_args["batch_size"],
    iterations=project_args["iterations"], 
    learning_rate=project_args["learning_rate"],
    weight_decay=project_args["weight_decay"],
    out_dir=Path(project_args["out_dir"]),
    device=device
):

    # Load Data 
    data_loader = fast_dataloader(
        inputs=pseudo_data, 
        targets=pseudo_target, 
        batch_size=batch_size,
        device=device
    )

    validation_data_loader = fast_dataloader(
        inputs=pseudo_data, 
        targets=pseudo_target, 
        batch_size=batch_size,
        device=device
    )

    # Load Model 
    model = WaveNet(n_ifos).to(device)

    val_method = Validation(validation_data_loader)

    training(
        iterations=iterations,
        data_loader=data_loader,
        model=model, 
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_func=val_method,
        out_dir=out_dir,
    )

    # Save Model  
    


if __name__ == "__main__":

    main()