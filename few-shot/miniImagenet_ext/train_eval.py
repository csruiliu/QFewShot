import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
import os
import pickle
import argparse

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
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # schedule learning rate to be cut in half every 2000 episodes
    scheduler = optim.lr_scheduler.StepLR(optimizer, config['decay_every'], gamma=0.5, last_epoch=-1)

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
            episode_dict = extract_episode(train_data['train_x'],
                                           train_data['train_y'],
                                           train_data['num_way'],
                                           train_data['num_shot'],
                                           train_data['num_query'])

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


def arg_config():
    parser = argparse.ArgumentParser(description='Train prototypical networks on miniImagenet')

    parser.add_argument('--model.learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--model.decay_every', type=int, default=20, metavar='LRDECAY',
                        help='number of epochs after which to decay the learning rate')
    parser.add_argument('--model.patience', type=int, default=200, metavar='PATIENCE',
                        help='number of epochs to wait before validation improvement (default: 1000)')

    parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--train.way', type=int, default=60, metavar='TRAINWAY',
                        help="number of classes per episode (default: 60) for training")
    parser.add_argument('--train.shot', type=int, default=5, metavar='TRIANSHOT',
                        help="number of support examples per class (default: 5) for training")
    parser.add_argument('--train.query', type=int, default=5, metavar='TRAINQUERY',
                        help="number of query examples per class (default: 5) for training")
    parser.add_argument('--train.episodes', type=int, default=100, metavar='NTRAIN',
                        help="number of train episodes per epoch (default: 100)")

    parser.add_argument('--eval.way', type=int, default=5, metavar='EVALWAY',
                        help="number of classes per episode in evaluation. 0 means same as train.way (default: 5)")
    parser.add_argument('--eval.shot', type=int, default=0, metavar='EVALSHOT',
                        help="number of support examples per class in evaluation. 0 means same as train.shot (default: 0)")
    parser.add_argument('--eval.query', type=int, default=15, metavar='EVALQUERY',
                        help="number of query examples per class in evaluation. 0 means same as train.query (default: 15)")
    parser.add_argument('--eval.episodes', type=int, default=100, metavar='NEVAL',
                        help="number of evaluation episodes per epoch (default: 100)")

    return vars(parser.parse_args())


def main():
    args = arg_config()

    model_learning_rate = args['model.learning_rate']
    model_decay_every = args['model.decay_every']
    model_patience = args['model.patience']

    train_epochs = args['train.epochs']
    train_way = args['train.way']
    train_shot = args['train.shot']
    train_query = args['train.query']
    train_episodes = args['train.episodes']

    eval_way = args['eval.way'] if args['eval.way'] != 0 else train_way
    eval_shot = args['eval.shot'] if args['eval.shot'] != 0 else train_shot
    eval_query = args['eval.query'] if args['eval.query'] != 0 else train_query
    eval_episodes = args['eval.episodes']


    MINIIMAGENET_DATA_DIR = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../../data/miniImagenet/data'))
    train_x, train_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR,  'train.pkl'))
    valid_x, valid_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR,  'valid.pkl'))

    results_dir = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../results'))

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    best_epoch = {
        'number': -1,
        'loss': np.inf,
        'acc': 0
    }

    config = {
        'results_dir': results_dir,
        'learning_rate': model_learning_rate,
        'decay_every': model_decay_every,
        'patience': model_patience,
        'best_epoch': best_epoch,
        'wait': 0,
        'stop': False
    }

    model = load_protonet(x_dim=(3, 84, 84), hid_dim=64, z_dim=64)

    train_data = {
        'train_x': train_x,
        'train_y': train_y,
        'num_way': train_way,
        'num_shot': train_shot,
        'num_query': train_query,
        'max_epoch': train_epochs,
        'epoch_size': train_episodes
    }

    valid_data = {
        'valid_x': valid_x,
        'valid_y': valid_y,
        'num_way': eval_way,
        'num_shot': eval_shot,
        'num_query': eval_query,
        'epoch_size': eval_episodes
    }

    train_logger = create_logger(os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../logs')), 'train.log')

    train(model, config, train_data, valid_data, train_logger)


if __name__ == "__main__":
    main()
