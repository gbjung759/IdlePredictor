import numpy as np
import random
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from custom_dataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
from stacked_lstm import StackedLSTM


class IdlePredictor:
    def __init__(self, train_path, test_path, optimizer, epochs, loss_function,
                 learning_rate, batch_size, early_stop_patience, seq_len, usecols):
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
            CustomDataset(path=train_path, seq_len=seq_len, use_cols=usecols),
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=2
        )
        self.test_dataloader = DataLoader(
            CustomDataset(path=test_path, seq_len=seq_len, use_cols=usecols),
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
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
        elif loss_ft == 'crossentropyloss':
            return torch.nn.CrossEntropyLoss()
        elif loss_ft == 'nllloss':
            return torch.nn.NLLLoss()
        else:
            raise ValueError('loss_function이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def predict(self):
        test_DataLoader = torch.utils.data.DataLoader(self.test_dataset, shuffle=False, batch_size=1)

        print("***** Running Prediction *****")
        print("  Num examples = {}".format(len(self.test_dataset)))
        print("  Test Batch size = 1")

        model.eval()
        pred = None
        label = None
        for batch in test_DataLoader:
            input_vector = batch[0].to(self.device)

            with torch.no_grad():
                predict = model(input_vector)

            if pred is None:
                pred = predict.detach().cpu().numpy()
                label = batch[1].numpy()
            else:
                pred = np.append(pred, predict.detach().cpu().numpy(), axis=0)
                label = np.append(label, batch[1].numpy(), axis=0)

        pred = np.argmax(pred, axis=1)

        print("***** Prediction 완료 *****")

        return pred.tolist(), label.tolist()



    def train(self):
        self.set_seed(42)
        train_iterator = trange(self.epochs, desc="Epoch")
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

    def load_model(self, path='model.pt'):
        with open(path, "rb") as f:
            dump = torch.load(f)
            self.model.load_state_dict(dump['state'])
        return dump

    @staticmethod
    def save_model(best_dict, loss_hist, path='model.pt'):
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
        plt.figure(figsize=(16, 4))
        plt.title("Training Loss Graph")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.plot(loss_hist)
        plt.savefig(path)
