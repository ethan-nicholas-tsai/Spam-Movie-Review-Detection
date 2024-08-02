import os

# Functions related to training data processing.
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class DataDeal:
    """
    Before starting training each time, initialize first.
    Calculate accuracy, precision, recall, and F1 based on the confusion matrix.
    Write the confusion matrix and results of a certain training session into a table.
    Save the model.
    Calculate the loss.
    """

    def __init__(self, tensorboard_log="./logs"):
        self.confusion_matrix_list = []  # Store the confusion matrix for each epoch round.
        self.acc_list = []
        self.recall_list = []
        self.F1_list = []
        self.precise_list = []

        self.recall_non_list = []
        self.F1_non_list = []
        self.precise_non_list = []

        self.train_loss_list = []
        self.val_loss_list = []
        self.test_loss_list = []

        # Record the maximal values
        self.acc_max = 0
        self.recall_max = 0
        self.F1_max = 0
        self.precise_max = 0

        # Whether use tensorboard or not
        if tensorboard_log:
            if not os.path.exists(tensorboard_log):
                os.makedirs(tensorboard_log)
            self.writer = SummaryWriter(tensorboard_log)

    def add_confusion_matrix_and_cal(self, confusion_matrix: torch):
        """
        tp[1,1]
        fp[1,0]
        tn[0,0]
        fn[0,1]
        :param confusion_matrix:
        :return:{"Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1": F1}
        """
        self.confusion_matrix_list.append(confusion_matrix)
        tp = confusion_matrix[1, 1]
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[1, 0]
        fn = confusion_matrix[0, 1]

        # Accuracy is the proportion of the total data that the model correctly identifies as true positives (TP) and true negatives (TN).
        accuracy = float((tp + tn) / (tp + tn + fn + fp))
        # Recall Rate: With respect to the positive class, the recall rate, also known as the true positive rate, is the proportion of actual positive cases that are correctly identified by the model.
        recall = float(tp / (tp + fn))
        precise = float(tp / (tp + fp))
        if precise + recall == 0:
            F1 = 0
        else:
            F1 = float(2 * precise * recall / (precise + recall))

        recall_non = float(tn / (tn + fp))
        precise_non = float(tn / (tn + fn))
        if precise_non + recall_non == 0:
            F1_non = 0
        else:
            F1_non = float(2 * precise_non * recall_non / (precise_non + recall_non))

        self.acc_list.append(accuracy)
        self.recall_list.append(recall)
        self.precise_list.append(precise)
        self.F1_list.append(F1)

        self.recall_non_list.append(recall_non)
        self.precise_non_list.append(precise_non)
        self.F1_non_list.append(F1_non)

        #
        if accuracy > self.acc_max:
            self.acc_max = accuracy
        if recall > self.recall_max:
            self.recall_max = recall
        if precise > self.precise_max:
            self.precise_max = precise
        if F1 > self.F1_max:
            self.F1_max = F1

        return {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precise,
            "F1": F1,
            # "Accuracy_non": accuracy,
            "Recall_non": recall_non,
            "Precision_non": precise_non,
            "F1_non": F1_non,
        }

    def write_confusion_matrix(self, save_path):
        """
        Write the training confusion matrix and results into a table.
        :param save_path:
        :return:
        """
        confusion_matrix_list = [
            i.resize(4).numpy().tolist() for i in self.confusion_matrix_list
        ]  # Convert the two-dimensional tensor of the confusion matrix into a one-dimensional list.
        confusion_matrix_list = torch.tensor(confusion_matrix_list)  # Convert into tensor
        tn = confusion_matrix_list[:, 0]
        fn = confusion_matrix_list[:, 1]
        fp = confusion_matrix_list[:, 2]
        tp = confusion_matrix_list[:, 3]

        df = pd.DataFrame(
            {
                "tn": tn.numpy().tolist(),
                "fn": fn.numpy().tolist(),
                "tp": tp.numpy().tolist(),
                "fp": fp.numpy().tolist(),
                "Accuracy": self.acc_list,
                "Precision": self.precise_list,
                "Recall": self.recall_list,
                "F1": self.F1_list,
            }
        )
        df.to_csv(save_path)

    def add_tensorboard_scalars(self, path, dir, epoch):
        """
        Multiple curves on a single chart.
        eg: writer.add_scalars(path, {'train loss': train_loss, 'val loss': val_loss}, epoch)
        :param path:
        :param dir:
        :param eopch:
        :return:
        """
        self.writer.add_scalars(path, dir, epoch)

    def add_tensorboard_scalar(self, path, data, epoch):
        # One curve per chart.
        self.writer.add_scalars(path, data, epoch)

    # Record training, validation, and testing losses.
    def add_train_loss(self, train_loss):
        self.train_loss_list.append(train_loss)

    def add_val_loss(self, val_loss):
        self.val_loss_list.append(val_loss)

    def add_test_loss(self, test_loss):
        self.test_loss_list.append(test_loss)


class DealConfusionMatrix:
    def __init__(self, save_path, write_all_flag=True):
        self.save_path = save_path
        self.write_all_flag = write_all_flag  # Write everything into the table at once.

    def write_matrix(self, data):
        """
        Write the confusion matrix
        :param data:
        :return:
        """
        if self.write_all_flag:
            df = pd.DataFrame(data, columns=["TP", "FP", "TN", "FN"])
            df.to_csv(self.save_path)
            return 0

        # Write line by line.
        if os.path.exists(self.save_path):
            df = pd.DataFrame(data)
            df.to_csv(self.save_path, mode="a", header=False, index=False)
        else:
            # If it hasn't been written before, just write a header into it.
            df = pd.DataFrame(data, columns=["TP", "FP", "TN", "FN"])
            df.to_csv(self.save_path, mode="a")


if __name__ == "__main__":
    pass
