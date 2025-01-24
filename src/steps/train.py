from src.utils.train_utils import HardUnetTrainer
from src.data.covid_dataset import CovidDataModule
from src.data.ISLES_dataset_2D import ISLESDataModule_2D
from src.models.FCHardnet import FCHardnet
from src.models.mseg_hardnet import HarDMSEG
from torch.nn import functional as F
import torch
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import BCELoss

pl.seed_everything(42, workers=True)

if __name__ == "__main__":
    params = yaml.safe_load(open("./src/params.yml"))

    model_type = params["train"]["model_type"]
    dataset_name = params["prepare-data"]["dataset"]
    modalities = params["prepare-data"]["modalities"]
    train_size = params["prepare-data"]["train_size"]
    roi_size_w = params["train"]["roi_size_w"]
    roi_size_h = params["train"]["roi_size_h"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]
    test_batch_size = params["train"]["test_batch_size"]
    opt = params["train"]["optimizer"]
    decay = params["train"]["decay"]
    momentum = params["train"]["momentum"]
    scheduler = params["train"]["scheduler"]
    loss_fn = params["train"]["loss"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(fr".\src\data\{dataset_name}_dataset.json", "r") as file:
        data = json.load(file)

    if dataset_name == 'covid':
        datamodule = CovidDataModule(
            batch_size=batch_size, device=device, data_properties=data
        )        
    elif dataset_name == 'ISLES' or dataset_name == 'isles':
        datamodule = ISLESDataModule_2D(data_properties=data, batch_size=batch_size, device=device, modalities=modalities)
    else:
        print('Unrecognized dataset')

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    

    print(
        f"-----------Total number of training batches: {len(train_loader)} ---------------------"
    )
    print(
        f"-----------Total number of validation batches: {len(val_loader)} ---------------------"
    )

    model = None
    if model_type == "FC":
        model = FCHardnet(n_classes=1, in_channels=1).to(device)
    elif model_type == "MSEG":
        arch = params["train"]["arch"]
        model = HarDMSEG(arch=arch).to(device)
    else:
        Exception(f"Unindentifiable architecture:{model_type}")

    optimizer = None
    if opt == "SGD":
        optimizer = torch.optim.SGD
    elif opt == "AdamW":
        optimizer = torch.optim.AdamW
    elif opt == "Adam":
        optimizer = torch.optim.Adam
    else:
        Exception(f"Unindentifiable optimizers:{opt}")

    if scheduler is not None:
        if scheduler == 'CosineAnnealingLR':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR
        elif scheduler == 'ConstantLR':
            sched = torch.optim.lr_scheduler.ConstantLR
        else:
            Exception(f'Unknown scheduler: {scheduler}')
        
    if loss_fn == 'diceloss':
        loss = DiceLoss
    elif loss_fn == 'diceceloss':
        loss = DiceCELoss
    elif loss_fn == 'bceloss':
        loss = BCELoss
    else:
        Exception(f'Unknown loss_fn: {loss_fn}')

    model = HardUnetTrainer(
        model=model,
        device=device,
        optim=optimizer,
        lr=lr,
        decay=decay,
        momentum=momentum,
        sched=sched,
        loss=loss
    )

    logger = TensorBoardLogger(
        "runs", name=f"{arch}_{dataset_name}_{lr}_{opt}_{decay}_{momentum}"
    )

    if scheduler is not None:
        logger.log_hyperparams(
            params=
                {"arch": arch,
                "train_size": train_size,
                "batch_size": batch_size,
                "lr": lr,
                "opt": opt,
                "decay": decay,
                "momentum": momentum,
                "scheduler": scheduler}
            ,
            metrics={},
        )
    else:
        logger.log_hyperparams(
            params=
                {"arch": arch,
                "train_size": train_size,
                "batch_size": batch_size,
                "lr": lr,
                "opt": opt,
                "decay": decay,
                "momentum": momentum}
            ,
            metrics={},
        )
    # initialize Lightning's trainer.
    max_epochs = params["train"]["max_epochs"]
    log_steps = params["train"]["log_steps"]
    check_val = params["train"]["check_val"]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=log_steps,
        check_val_every_n_epoch=check_val,
        logger=logger,
    )

    # train
    print(f"----------- Starting Training ---------------------")
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(f"./weights/{arch}_{dataset_name}_{lr}_{opt}_{decay}_{momentum}.pth")
