import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import fsum
from tqdm import trange
import os
import pickle

from data_loader import load_images, extract_episode
from few_shot import load_protonet
from logger import create_logger


# model -> model structure
# config -> dict with some hyperparameters and important configs
# valid_data -> validation set, num_way, num_shot, num_query etc
# epoch -> number of the respective training epoch
def evaluate_valid(model, config, valid_data, curr_epoch, logger):
    # set model to evaluation mode
    model.eval()

    valid_loss = 0.0
    valid_acc = 0.0

    logger.info('> Validation')

    # do epoch_size classification tasks to evaluate the model
    for episode in trange(valid_data['epoch_size']):
        # get the episode dict
        episode_dict = extract_episode(
            valid_data['valid_x'], valid_data['valid_y'], valid_data['num_way'],
            valid_data['num_shot'], valid_data['num_query'])

        # classify images and get the loss and the acc of the curr episode
        _, output = model.set_forward_loss(episode_dict)

        # acumulate the loss and the acc
        valid_loss += output['loss']
        valid_acc += output['acc']

    # average the loss and the acc to get the valid loss and the acc
    valid_loss = valid_loss / valid_data['epoch_size']
    valid_acc = valid_acc / valid_data['epoch_size']

    # output the valid loss and the valid acc
    logger.info('Loss: %.4f / Acc: %.2f%%' % (valid_loss, (valid_acc * 100)))

    # implement early stopping mechanism
    # check if valid_loss is the best so far
    if config['best_epoch']['loss'] > valid_loss:
        # if true, save the respective train epoch
        config['best_epoch']['number'] = curr_epoch

        # save the best loss and the respective acc
        config['best_epoch']['loss'] = valid_loss
        config['best_epoch']['acc'] = valid_acc

        # save the model with the best loss so far
        model_file = os.path.join(config['results_dir'], 'best_model.pth')
        torch.save(model.state_dict(), model_file)

        logger.info('=> This is the best model so far! Saving...')

        # set wait to zero
        config['wait'] = 0
    else:
        # if false, increment the wait
        config['wait'] += 1

        # when the wait is bigger than the patience
        if config['wait'] > config['patience']:
            # the train has to stop
            config['stop'] = True

            logger.info('Patience was exceeded... Stopping...')


# model -> model structure
# config -> dict with some hyperparameters and important configs
# train_data -> train set, num_way, num_shot, num_query etc
# valid_data -> validation set, num_way, num_shot, num_query etc
def train(model, config, train_data, valid_data, logger):
    # set Adam optimizer with an initial learning rate
    optimizer = optim.Adam(
        model.parameters(), lr = config['learning_rate'])

    # schedule learning rate to be cut in half every 2000 episodes
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config['decay_every'], gamma = 0.5, last_epoch = -1)

    # set model to training mode
    model.train()

    # number of epochs so far
    epochs_so_far = 0

    # train until early stopping says so
    # or until the max number of epochs is not achived
    while epochs_so_far < train_data['max_epoch'] and not config['stop']:
        epoch_loss = 0.0
        epoch_acc = 0.0

        logger.info('==> Epoch %d' % (epochs_so_far + 1))

        logger.info('> Training')

        # do epoch_size classification tasks to train the model
        for episode in trange(train_data['epoch_size']):
            # get the episode dict
            episode_dict = extract_episode(
              train_data['train_x'], train_data['train_y'], train_data['num_way'],
              train_data['num_shot'], train_data['num_query'])

            optimizer.zero_grad()

            # classify images and get the loss and the acc of the curr episode
            loss, output = model.set_forward_loss(episode_dict)

            # acumulate the loss and the acc
            epoch_loss += output['loss']
            epoch_acc += output['acc']

            # update the model parameters (weights and biases)
            loss.backward()
            optimizer.step()

        # average the loss and the acc to get the epoch loss and the acc
        epoch_loss = epoch_loss / train_data['epoch_size']
        epoch_acc = epoch_acc / train_data['epoch_size']

        # output the epoch loss and the epoch acc
        logger.info('Loss: %.4f / Acc: %.2f%%' % (epoch_loss, (epoch_acc * 100)))

        # do one epoch of evaluation on the validation test
        evaluate_valid(model, config, valid_data, epochs_so_far + 1, logger)

        # increment the number of epochs
        epochs_so_far += 1

        # tell the scheduler to increment its counter
        scheduler.step()

    # get dict with info about the best epoch
    best_epoch = config['best_epoch']

    # at the end of the training, output the best loss and the best acc
    logger.info('Best loss: %.4f / Best Acc: %.2f%%'
          % (best_epoch['loss'], (best_epoch['acc'] * 100)))

    # save dict with info about the best epoch
    with open(os.path.join(config['results_dir'], 'best_epoch.pkl'), 'wb') as f:
        pickle.dump(best_epoch, f, pickle.HIGHEST_PROTOCOL)


def main():
    dataset = "mini_imagenet"

    MINIIMAGENET_DATA_DIR = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../../data/miniImagenet/data'))
    print(MINIIMAGENET_DATA_DIR)
    train_x, train_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR,  'train.pkl'))
    valid_x, valid_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR,  'valid.pkl'))
    # test_x, test_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR,  'test.pkl'))

    # episode_dict = extract_episode(train_x, train_y, num_way=5, num_shot=5, num_query=5)

    results_dir = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../results'))

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    best_epoch = {
        'number': -1,
        'loss': np.inf,
        'acc': 0}

    config = {
        'results_dir': results_dir,
        'learning_rate': 0.001,
        'decay_every': 20,
        'patience': 200,
        'best_epoch': best_epoch,
        'wait': 0,
        'stop': False}

    model = load_protonet(x_dim=(3, 84, 84), hid_dim=64, z_dim=64)

    train_data = {
        'train_x': train_x,
        'train_y': train_y,
        'num_way': 20,
        'num_shot': 5,
        'num_query': 15,
        'max_epoch': 10000,
        'epoch_size': 100}

    valid_data = {
        'valid_x': valid_x,
        'valid_y': valid_y,
        'num_way': 5,
        'num_shot': 5,
        'num_query': 15,
        'epoch_size': 100}

    train_logger = create_logger(os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../logs')), 'train.log')

    train(model, config, train_data, valid_data, train_logger)


if __name__ == "__main__":
    main()
