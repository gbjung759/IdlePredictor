import json
import torch
from idle_predictor import IdlePredictor
from stacked_lstm import StackedLSTM


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
        batch_size = config['batch_size']
        loss_function = config['loss_function']
        learning_rate = config['learning_rate']
        early_stop_patience = config['early_stop_patience']
        optimizer = config['optimizer']
        epochs = config['epochs']
        seq_len = config['seq_len']
        usecols = ['Timestamp', 'Response', 'IOType', 'Offset', 'Size']

    idle_predictor = IdlePredictor(train_path='./dataset/train',
                                   test_path='./dataset/test',
                                   optimizer=optimizer,
                                   epochs=epochs,
                                   learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   early_stop_patience=early_stop_patience,
                                   loss_function=loss_function,
                                   seq_len=seq_len,
                                   usecols=usecols)
    try:
        dump = idle_predictor.load_model()
        loss_hist = dump['loss_history']
        print('skip train..')
    except:
        idle_predictor.train()
        best_dict, loss_hist = idle_predictor.train()
        idle_predictor.save_model(best_dict=best_dict, loss_hist=loss_hist)
        idle_predictor.load_model()
        idle_predictor.save_loss_hist(loss_hist)
        print('model trained and saved..')
        pass

