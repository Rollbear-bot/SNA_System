# -*- coding: utf-8 -*-
# @Time: 2020/12/14 8:23
# @Author: Rollbear
# @Filename: timer.py

from functools import wraps
from time import time


def timer(debug):
    """计时器"""
    def _timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if debug:
                print("-"*20)
                print(f"in \"{func.__name__}\"...")
            start = time()
            func_res = func(*args, **kwargs)
            # 打印函数执行用时
            if debug:
                print(f"exec \"{func.__name__}\" in {time() - start}s.")
            return func_res  # 返回函数运行结果
        return wrapper
    return _timer
