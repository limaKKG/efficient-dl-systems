import torch
from torch import nn
from tqdm.auto import tqdm
from unet import Unet
from dataset import get_train_data
from scaler import ManualLossScaler
from typing import Optional

def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[ManualLossScaler] = None,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            loss = scaler.scale_loss(loss).backward()

            if not scaler._has_inf_or_nan(model):
                scaler.unscale_(optimizer)
                optimizer.step()
            scaler.update(scaler._has_inf_or_nan(model))


        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train():

    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    scaler = ManualLossScaler(init_scale=1024.0, dynamic=False)
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)


