import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s] [%(asctime)s] %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S")

logger = logging.getLogger()
fh = logging.FileHandler(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'logs/log_test.txt')))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

import idne


