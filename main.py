import logging, globals
from util import load_mnist_X, load_mnist_Y, load_regression_X, load_regression_Y, z_norm

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    config_options = globals.config
    task_path = config_options.get("Data", dataset)

    if dataset == "classify":
        Xtrain = z_norm(load_mnist_X(task_path + "classf_Xtrain.txt"))
        Xtest = z_norm(load_mnist_X(task_path + "classf_Xtest.txt"))
        Xval = z_norm(load_mnist_X(task_path + "classf_XVal.txt"))
        ytrain = load_mnist_Y(task_path + "classf_ytrain.txt")
        ytest = load_mnist_Y(task_path + "classf_ytest.txt")
        yval = load_mnist_Y(task_path + "classf_yVal.txt")
    elif dataset == "regression":
        Xtrain = z_norm(load_regression_X(task_path + "regr_Xtrain.txt"))
        Xtest = z_norm(load_regression_X(task_path + "regr_Xtest.txt"))
        Xval = z_norm(load_regression_X(task_path + "regr_Xval.txt"))
        ytrain = load_regression_Y(task_path + "regr_ytrain.txt")
        ytest = load_regression_Y(task_path + "regr_ytest.txt")
        yval = load_regression_Y(task_path + "regr_yval.txt")
    else:
        logger.info("Invalid task.")

def test(dataset):
    pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train memory network')
    train_parser.add_argument('dataset',
                              help='The dataset to train.')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test memory network')
    test_parser.add_argument('dataset',
                             help='The dataset to test')
    test_parser.set_defaults(which='test')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)


if __name__ == '__main__':
    main()