

import torch
from torch import nn
from torch.optim import Adam



def training(
    iterations: int, 
    data_loader, 
    model: nn.Module, 
    learning_rate,
    weight_decay,
    criterion = nn.CrossEntropyLoss(), 
    val_func = None,
    out_dir= None
):

    opt = Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    for iter in range(iterations):

        for epoch, (x, y) in enumerate(data_loader):

            p_value = model(x)
            loss = criterion(p_value, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_func is not None:

            val_func(model)

    if out_dir is not None:
        torch.save(model.state_dict(), out_dir/f"final_model.pt")

class Validation:

    def __init__(
        self,
        dataloader
    ):
        
        self.dataloader=dataloader


    @torch.no_grad()
    def __call__(self, model):

        model.eval()
        for epoch_v, (x, y) in enumerate(self.dataloader):

            p_value = model(x)

        model.train()