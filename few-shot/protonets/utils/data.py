import protonets.data


def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)
    elif opt['data.dataset'] == 'miniImagenet':
        ds = protonets.data.miniImagenet.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
