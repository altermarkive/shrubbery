#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import List

from numerapi import NumerAPI

from shrubbery.observability import logger


def numerai_api() -> NumerAPI:
    public_id = os.environ.get('NUMERAI_PUBLIC_ID')
    secret_key = os.environ.get('NUMERAI_SECRET_KEY')
    while True:
        try:
            napi = NumerAPI(public_id=public_id, secret_key=secret_key)
            return napi
        except Exception:
            logger.exception('Login failed')
            time.sleep(10)


def numerai_models() -> List[str]:
    return list(napi.get_models().keys())


napi = numerai_api()
