# #! /usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------
#!  Copyright(C) 2020
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:2020010311
#   作   者：陈瑞侠
#   完成日期：2020-1-3
# -----------------------------

import logging
import logging.config
from os import path
import os
import json
import logging

# def get_logger():
#     log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logger_config.ini')
#     logging.config.fileConfig(log_file_path)
#     return logging.getLogger()
# bz_log = get_logger()



def setup_logging(default_path=path.join(path.dirname(path.abspath(__file__)),
                  "logger_config.json"), default_level=logging.INFO):
    if os.path.exists(default_path):
        with open(default_path,"r") as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return logging.getLogger()

bz_log = setup_logging()