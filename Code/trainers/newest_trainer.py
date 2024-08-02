import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)
import copy
from utils.train_util import EarlyStopping
from experiments.backend.utils.result_record import DataDeal

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def train(
    model,
    train_loader,
    val_loader,
    epoches,
    lr,
    momentum,
    weight_decay,
    cos_train,
    early_stop=False,
    patience=20,
    info="",
    save_dir="./",
):
    print(info)

    # Run on which GPU
    device = model.device
    print(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Build optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if cos_train:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)

    # Early Stop
    if early_stop:
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=os.path.join(save_dir, "checkpoint.pt"),
        )

    # Log losses and confusion matrix
    data_deal = DataDeal()

    # Save the best model
    min_loss_val = 10 
    min_epoch = 1 
    best_model = None
    best_model_name = None

    # Train
    for epoch in range(1, epoches + 1):
        model.train()
        pbar = tqdm(train_loader)  # Progress bar
        epoch_loss = []
        for labels, movie_ids, text_features, review_ids in pbar:
            optimizer.zero_grad()
            model.zero_grad()
            labels, movie_ids = (
                labels.to(device),
                movie_ids.to(device),
            )
            text_features = [it.to(device) for it in text_features]

            output = model(text_features, review_ids, movie_ids, train=True)

            loss = criterion(output, labels)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(f"epoch{epoch} loss:{sum(epoch_loss) / len(pbar)}")

        train_loss = sum(epoch_loss) / len(pbar)
        data_deal.add_train_loss(train_loss)
        pbar.close()
        # print(f"train：epoch{epoch} loss{train_loss}")
        if cos_train:
            scheduler.step() 
        cur_lr = optimizer.param_groups[-1]["lr"]
        # cur_lr_list.append(cur_lr)
        print("cur_lr:", cur_lr)

        # val
        with torch.no_grad():
            model.eval()
            pbar = tqdm(val_loader)
            val_epoch_loss = []
            confusion_matrix = torch.zeros(2, 2)

            for labels, movie_ids, text_features, review_ids in pbar:
                labels, movie_ids = (
                    labels.to(device),
                    movie_ids.to(device),
                )
                text_features = [it.to(device) for it in text_features]

                output = model(text_features, review_ids, movie_ids, train=False)

                loss = criterion(output, labels)

                val_epoch_loss.append(loss.item())

                result = output.argmax(1)
                for t, p in zip(result.view(-1), labels.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            score = data_deal.add_confusion_matrix_and_cal(confusion_matrix)
            val_loss = sum(val_epoch_loss) / len(pbar)

            data_deal.add_val_loss(val_loss)
            pbar.close()
            print(
                f"test: epoch{epoch} loss{val_loss} acc{score['Accuracy']} recall{score['Recall']} F1 {score['F1']}"
            )
            # Save model
            if epoch > min_epoch and val_loss < min_loss_val:
                min_loss_val = val_loss
                best_model = copy.deepcopy(model)
                best_model_name = f"{epoch}-loss{val_loss}-F1{score['F1']}-acc{score['Accuracy']}-recall{score['Recall']}-pre{score['Precision']}.pt"

            # Early Stop
            if early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # Plot loss curve
        data_deal.add_tensorboard_scalars(
            f"{info}-epoch{epoches}-lr{lr}/loss",
            {"train loss": train_loss, "val loss": val_loss},
            epoch,
        )
        # Plot confusion matrix
        data_deal.add_tensorboard_scalars(
            f"{info}-epoch{epoches}-lr{lr}/confusion_matrix",
            {
                "[0,0]": confusion_matrix[0][0],
                "[0,1]": confusion_matrix[0][1],
                "[1,0]": confusion_matrix[1][0],
                "[1,1]": confusion_matrix[1][1],
            },
            epoch,
        )
        # Plot metric curves
        data_deal.add_tensorboard_scalars(
            f"{info}-epoch{epoches}-lr{lr}/result",
            {
                "acc": score["Accuracy"],
                "recall": score["Recall"],
                "pre": score["Precision"],
                "F1": score["F1"],
            },
            epoch,
        )

        data_deal.add_tensorboard_scalars(
            f"{info}-epoch{epoches}-lr{lr}/result_non",
            {
                "acc": score["Accuracy"],
                "recall": score["Recall_non"],
                "pre": score["Precision_non"],
                "F1": score["F1_non"],
            },
            epoch,
        )
    # Save the best model
    model_save_dir = os.path.join(save_dir, info)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, best_model_name)
    torch.save(best_model, model_save_path)
    # Save the result on validation set
    data_deal.write_confusion_matrix(os.path.join(save_dir, "result_val.csv"))
