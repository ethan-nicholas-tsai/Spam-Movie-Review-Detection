import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from experiments.backend.utils.result_record import DataDeal
from experiments.backend.utils.bases.track_base import Timer

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def predict(
    model,
    test_loader,
    info="",
    save_dir="./",
):
    print(info)
    # Run on which GPU
    device = model.device
    # Log losses and confusion matrix
    data_deal = DataDeal()
    # Begin timer
    timer = Timer()
    timer.start()
    # test
    with torch.no_grad():
        pbar = tqdm(test_loader)
        confusion_matrix = torch.zeros(2, 2)

        for labels, movie_ids, text_features, review_ids in pbar:
            labels, movie_ids = (
                labels.to(device),
                movie_ids.to(device),
            )
            text_features = [it.to(device) for it in text_features]

            output = model(text_features, review_ids, movie_ids, train=False)

            result = output.argmax(1)
            for t, p in zip(result.view(-1), labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        score = data_deal.add_confusion_matrix_and_cal(confusion_matrix)

        pbar.close()
    time_span = timer.stop()
    print(score)
    # Save the result on test set
    data_deal.write_confusion_matrix(os.path.join(save_dir, "result_test.csv"))
    with open(os.path.join(save_dir, "time.txt"), "w+") as f:
        f.write(str(time_span))
