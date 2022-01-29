import torch
import os
import argparse
from tqdm import trange
from math import fsum

from data_loader import load_images, extract_episode
from logger import create_logger
from few_shot import load_protonet


# model -> model structure
# test_data -> test set, num_way, num_shot, num_query etc
def evaluate_test(model, config, test_data, logger):
    # load the saved model
    state_dict = torch.load(os.path.join(config['results_dir'], 'best_model.pth'))
    model.load_state_dict(state_dict)

    # set the model to evaluation mode
    model.eval()

    test_loss = 0.0
    test_acc = []

    logger.info('> Testing')

    # do epoch_size classification tasks to test the model
    for episode in trange(test_data['epoch_size']):
        # get the episode_dict
        episode_dict = extract_episode(test_data['test_x'],
                                       test_data['test_y'],
                                       test_data['num_way'],
                                       test_data['num_shot'],
                                       test_data['num_query'])

        # classify images and get the loss and the acc of the curr episode
        _, output = model.set_forward_loss(episode_dict)

        # acumulate the loss and the acc
        test_loss += output['loss']
        test_acc.append(output['acc'])

    # average the loss
    test_loss = test_loss / test_data['epoch_size']

    # average the acc
    test_acc_avg = sum(test_acc) / test_data['epoch_size']

    # calculate the standard deviation
    test_acc_dev = fsum([((x - test_acc_avg) ** 2) for x in test_acc])
    test_acc_dev = (test_acc_dev / (test_data['epoch_size'] - 1)) ** 0.5

    # calculate error considering 95% confidence interval
    error = 1.96 * test_acc_dev / (test_data['epoch_size'] ** 0.5)

    # output the test loss and the test acc
    logger.info('Loss: %.4f / Acc: %.2f +/- %.2f%%' % (test_loss, test_acc_avg * 100, error * 100))

    return test_acc_avg


def run_evaluation_n_times(n, model, config, test_data, logger):
    test_acc_list = []

    test_acc = 0
    std_dev = 0

    for i in range(n):
        output = evaluate_test(model, config, test_data, logger)

        test_acc_list.append(output)
        test_acc += output

    # standard deviation
    test_acc = test_acc / n

    # standard deviation
    std_dev = fsum([((x - test_acc) ** 2) for x in test_acc_list])
    std_dev = (std_dev / (n - 1)) ** 0.5

    # calculate error considering 95% confidence interval
    error = 1.96 * std_dev / (n ** 0.5)

    # output the test loss and the test acc
    logger.info('With %i run(s), Acc: %.2f +/- %.2f%%' % (n, test_acc * 100, error * 100))


def arg_config():
    parser = argparse.ArgumentParser(description='Test prototypical networks on miniImagenet')

    parser.add_argument('--model.learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--model.decay_every', type=int, default=20, metavar='LRDECAY',
                        help='number of epochs after which to decay the learning rate')
    parser.add_argument('--model.patience', type=int, default=200, metavar='PATIENCE',
                        help='number of epochs to wait before validation improvement (default: 1000)')

    parser.add_argument('--test.way', type=int, default=5, metavar='TESTWAY',
                        help="number of classes per episode in test. (default: 5)")
    parser.add_argument('--test.shot', type=int, default=5, metavar='TESTSHOT',
                        help="number of support examples per class in test. (default: 5)")
    parser.add_argument('--test.query', type=int, default=15, metavar='TESTQUERY',
                        help="number of query examples per class in test. (default: 15)")
    parser.add_argument('--test.episodes', type=int, default=600, metavar='NTEST',
                        help="number of evaluation episodes per epoch (default: 600)")

    parser.add_argument('--test.runs', type=int, default=3, metavar='NRUNS',
                        help="number of testing rounds (default: 3)")

    return vars(parser.parse_args())


def main():
    args = arg_config()

    model_learning_rate = args['model.learning_rate']
    model_decay_every = args['model.decay_every']
    model_patience = args['model.patience']

    test_way = args['test.way']
    test_shot = args['test.shot']
    test_query = args['test.query']
    test_episodes = args['test.episodes']
    test_runs = args['test.runs']

    results_dir = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../results'))

    config = {
        'results_dir': results_dir,
        'learning_rate': model_learning_rate,
        'decay_every': model_decay_every,
        'patience': model_patience,
        'wait': 0,
        'stop': False}

    model = load_protonet(x_dim=(3, 84, 84), hid_dim=64, z_dim=64)

    MINIIMAGENET_DATA_DIR = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../../data/miniImagenet/data'))
    test_x, test_y = load_images(os.path.join(MINIIMAGENET_DATA_DIR, 'test.pkl'))

    test_data = {
        'test_x': test_x,
        'test_y': test_y,
        'num_way': test_way,
        'num_shot': test_shot,
        'num_query': test_query,
        'epoch_size': test_episodes
    }

    test_logger = create_logger(os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../logs')), 'test.log')

    run_evaluation_n_times(test_runs, model, config, test_data, test_logger)


if __name__ == "__main__":
    main()
