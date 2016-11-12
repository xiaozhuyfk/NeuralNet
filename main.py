import logging, globals
from util import load_mnist_X, load_mnist_Y, load_regression_X, load_regression_Y, z_norm, writeFile
from model import Model, Layer
from activations import Activation

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    config_options = globals.config
    task_path = config_options.get("Data", dataset)
    loss = config_options.get('Train', 'loss')
    activation = config_options.get('Train', 'activation')

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
        logger.warning("Invalid task.")
        return
    logger.info("Load data complete.")

    # build model
    N, input_dim = Xtrain.shape
    model = Model()
    model.add(
        Layer(output_dim=globals.layer_dim,
              input_dim=input_dim)
    )
    model.add(
        Layer(output_dim=globals.output_dim)
    )
    model.add(
        Activation(activation=activation)
    )

    model.compile(loss=loss)
    history = model.fit(Xtrain, ytrain,
                        batch_size=N,
                        iterations=globals.iterations,
                        validation_data=(Xval, yval))

    # save result
    result_dir = config_options.get('Result', 'result-dir')
    file_name = "_".join([dataset,
                          activation,
                          str(globals.alpha),
                          str(globals.lam),
                          str(globals.layer_dim),
                          str(globals.iterations)]) + ".txt"
    file_path = result_dir + file_name
    writeFile(file_path, "")
    for datum in history:
        datum = [str(x) for x in datum]
        line = "\t".join(datum) + "\n"
        writeFile(file_path, line, 'a')


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