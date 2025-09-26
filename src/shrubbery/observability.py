#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


def setup_logger() -> logging.Logger:
    logging_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=logging_format, level=logging.INFO)
    logging.getLogger('distributed.core').setLevel(logging.WARNING)
    logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
    logging.getLogger('distributed.nanny').setLevel(logging.WARNING)
    logging.getLogger('distributed.http.proxy').setLevel(logging.WARNING)
    logger = logging.getLogger('shrubbery')
    return logger


logger = setup_logger()
