#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


def setup_logger() -> logging.Logger:
    logging_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=logging_format, level=logging.INFO)
    logger = logging.getLogger('shrubbery')
    return logger


logger = setup_logger()
