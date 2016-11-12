from ConfigParser import SafeConfigParser
import logging

logger = logging.getLogger(__name__)

config = None

alpha = 0.0
lam = 0.0
layer_dim = 0
output_dim = 0
iterations = 0

def read_configuration(configfile):
    global config
    global alpha, lam, output_dim, iterations, layer_dim

    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser

    alpha = float(config.get('Train', 'alpha'))
    lam = float(config.get('Train', 'lambda'))
    layer_dim = int(config.get('Train', 'layer-dim'))
    output_dim = int(config.get('Train', 'output-dim'))
    iterations = int(config.get('Train', 'iterations'))

    return parser
