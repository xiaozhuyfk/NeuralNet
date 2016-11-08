from ConfigParser import SafeConfigParser
import logging

logger = logging.getLogger(__name__)

config = None

def read_configuration(configfile):
    global config
    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser
    return parser