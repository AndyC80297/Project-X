from torch.utils.data import Dataset, DataLoader, TensorDataset



def fast_dataloader(
    inputs,
    targets,
    shuffle=True,
    batch_size = 32,
    pin_memory=True, 
    device="cpu"
):

    dataset = TensorDataset(
        inputs.to(device), 
        targets.to(device)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)