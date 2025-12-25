# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/25 10:22
# FileName: plug-in
# Project: RemoteFaceRecognitionService
# Author: JasonLai
# Email: jasonlaihj@163.com
import logging
import os


def log_func(root_path):
    """
    项目运行日志
    :param root_path:项目根路径
    :return:
    """
    log_name = 'py_log'  # 日志名称(默认py_log)
    log_level = logging.DEBUG  # 日志等级(默认DEBUG)
    console = True  # 是否在控制台输出(默认在控制台输出)
    # 日志输出格式
    log_format = u'[%(asctime)s] - [%(filename)s :%(lineno)d line] - %(levelname)s: %(message)s'
    # 日志文件的名称和路径
    log_file_path = os.path.join(root_path, "log\\" + 'py_log_file.txt')
    # 日志文件夹
    log_dir_path = os.path.dirname(log_file_path)

    if not os.path.exists(log_dir_path):  # 判断文件夹是否存在
        os.makedirs(log_dir_path)  # 创建日志文件夹

    py_loges = logging.getLogger(log_name)  # 创建log
    py_loges.setLevel(log_level)  # 设置等级

    # 定义log格式
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(log_file_path)  # 创建文件
    file_handler.setFormatter(formatter)

    # 遍历多个日志级别
    for level_content in [logging.DEBUG, logging.ERROR, logging.INFO]:
        file_handler.setLevel(level_content)  # 设置写入日志文件的内容

    py_loges.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()  # 创建控制台
        console_handler.setFormatter(formatter)
        py_loges.addHandler(console_handler)

    with open(log_file_path, "a+") as f:
        f.truncate()

    return py_loges


# project_path = os.path.abspath(os.getcwd())

# 创建日志对象
logger = log_func("./")
