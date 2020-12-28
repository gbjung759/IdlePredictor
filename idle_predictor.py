import numpy as np
import random
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from custom_dataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
from stacked_lstm import StackedLSTM
from sklearn.metrics import *


class IdlePredictor:
    def __init__(self, train_path, test_path, optimizer, epochs, loss_function,
                 learning_rate, batch_size, early_stop_patience, seq_len, usecols, test_set_ratio):
        self.optimizer = optimizer.lower()
        assert type(self.optimizer) is str, 'optimizer_name의 type은 string이 되어야 합니다.'
        self.loss_function = loss_function
        assert type(loss_function) is str, 'loss_ft type은 string이 되어야 합니다.'
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.seq_len = seq_len
        self.train_dataloader = DataLoader(
            CustomDataset(path=train_path, seq_len=seq_len, use_cols=usecols, test_set_ratio=test_set_ratio),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )
        self.test_dataloader = DataLoader(
            CustomDataset(path=test_path, seq_len=seq_len, use_cols=usecols, test_set_ratio=test_set_ratio),
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StackedLSTM(input_dim=len(usecols)).to(self.device)
        self.optimizer = self.get_optimizer(self.model.parameters(), self.optimizer, self.learning_rate)
        self.loss_function = self.get_loss_function(self.loss_function)

    @staticmethod
    def set_seed(seed):
        """
        for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_optimizer(self, parameters, optimizer_name, learning_rate):
        if optimizer_name == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(parameters, lr=learning_rate)
        else:
            raise ValueError('optimizer_name이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def get_loss_function(self, loss_ft):
        if loss_ft == 'mseloss':
            return torch.nn.MSELoss()
        elif loss_ft == 'l1loss':
            return torch.nn.L1Loss()
        elif loss_ft == 'crossentropyloss':
            return torch.nn.CrossEntropyLoss()
        elif loss_ft == 'nllloss':
            return torch.nn.NLLLoss()
        else:
            raise ValueError('loss_function이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def train(self):
        self.set_seed(42)
        train_iterator = trange(self.epochs, desc="Epoch")
        print("\n***** Running training *****")
        print("  Num Epochs = {}".format(self.epochs))
        print("  Train Batch size = {}".format(self.batch_size))
        print("  Device = ", self.device)
        loss_history = []
        self.model.to(self.device)
        self.model.train(True)
        self.model.zero_grad()
        best = {"loss": sys.float_info.max, "anger": 0}
        for epoch in train_iterator:
            loss_in_epoch = 0.0
            for j, [input_vector, label] in enumerate(self.train_dataloader):
                input_vector = input_vector.to(self.device)
                output = self.model(input_vector)
                label = label.to(self.device)
                criterion = self.loss_function(output, label)
                criterion.backward()
                # self.optimizer.zero_grad()
                self.optimizer.step()
                self.model.zero_grad()
                loss_in_epoch += criterion.item()
            loss_history.append(loss_in_epoch)
            if loss_in_epoch < best['loss']:
                best["state"] = self.model.state_dict()
                best["loss"] = loss_in_epoch
                best["epoch"] = epoch + 1
                best['anger'] = 0
            else:
                best['anger'] += 1
            if best['anger'] > self.early_stop_patience:
                break
            if (epoch + 1) % 1 == 0:
                print("  Epoch / Total Epoch : {} / {}".format(epoch + 1, self.epochs))
                print("  Loss : {:.4f}".format(loss_in_epoch))
        self.model.train(False)
        return best, loss_history

    def predict(self):
        print("***** Running Prediction *****")
        print("  Test Batch size = 1")
        self.model.eval()
        predictions = None
        labels = None
        for [input_vector, label] in self.test_dataloader:
            input_vector = input_vector.to(self.device)
            with torch.no_grad():
                output = self.model(input_vector)
            if predictions is None:
                predictions = output.detach().cpu().numpy()
                labels = label.numpy()
            else:
                predictions = np.append(predictions, output.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, label.numpy(), axis=0)
        return self.get_idle_from_timestamp(predictions.tolist()), self.get_idle_from_timestamp(labels.tolist())

    def eval(self, pred, label):
        result = dict()
        result['mean_absolute_error'] = mean_absolute_error(y_true=label, y_pred=pred)
        result['mean_squared_error'] = mean_squared_error(y_true=label, y_pred=pred, squared=True)
        result['root_mean_squared_error'] = mean_squared_error(y_true=label, y_pred=pred, squared=False)
        result['mean_idle_time'] = np.mean(y_true)
        result['r2_score'] = r2_score(y_true=label, y_pred=pred)
        return result

    def load_model(self, path='./model/model.pt'):
        with open(path, "rb") as f:
            dump = torch.load(f)
            self.model.load_state_dict(dump['state'])
        return dump

    @staticmethod
    def save_model(best_dict, loss_hist, path='./model/model.pt'):
        with open(path, "wb") as f:
            torch.save(
                {
                    "state": best_dict["state"],
                    "best_epoch": best_dict["epoch"],
                    "loss_history": loss_hist,
                },
                f,
            )

    @staticmethod
    def save_loss_hist(loss_hist, path='train_loss'):
        plt.plot(np.arange(1, len(loss_hist) + 1), loss_hist, 'k-', label='Train loss', linewidth=1)
        plt.xlabel('Epochs', labelpad=10, fontsize=18)
        plt.ylabel('Train Loss', labelpad=10, fontsize=18)
        plt.xlim(1, len(loss_hist) + 1)
        plt.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1.0, fontsize=16)
        plt.grid(color='gray', dashes=(2,2))
        plt.savefig(path)

    @staticmethod
    def plot_prediction(pred, label, path='prediction', max_len=10000):
        if len(pred) > max_len:
            start_index = random.randint(0, len(pred) - max_len)
            end_index = start_index + max_len
            pred = pred[start_index:end_index]
            label = label[start_index:end_index]
        pred = np.asarray(pred)
        label = np.asarray(label)
        plt.plot(np.arange(1, len(pred) + 1), pred, 'r-', label='Predicted idle time', linewidth=1)
        plt.plot(np.arange(1, len(label) + 1), label, 'b-', label='Real idle time', linewidth=1)
        plt.xlim(1, len(pred) + 1)
        plt.ylim(np.min(np.append(pred, label)), np.max(np.append(pred, label)))
        plt.xlabel('I/O commands', labelpad=10, fontsize=18)
        plt.ylabel('Idle time between I/O commands', labelpad=10, fontsize=18)
        plt.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1.0, fontsize=16)
        plt.grid(color='gray', dashes=(2,2))
        plt.show()
        plt.savefig(path)

    @staticmethod
    def get_idle_from_timestamp(timestamps):
        idletimes = list()
        for i in range(len(timestamps) - 1):
            idletimes.append(timestamps[i+1] - timestamps[i])
        return idletimes
