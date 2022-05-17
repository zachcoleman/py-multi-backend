import numpy as np
from python.py_rle import py_mask_to_rle
from cython_rle import cy_mask_to_rle
from py_multi_backend import thread_f_order_mask2rle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time

SAMP_SIZE = 100
SIZE = (3072, 3072)
rand_mask = np.random.random(SIZE) < 0.5
static_mask = np.ones(SIZE) < 0.5

if __name__ == "__main__":
    a = time.perf_counter()
    with ThreadPoolExecutor(1) as ex:
        for _ in range(SAMP_SIZE):
            ex.submit(thread_f_order_mask2rle, rand_mask)
    print(time.perf_counter()-a)

    a = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        for _ in range(SAMP_SIZE):
            ex.submit(thread_f_order_mask2rle, rand_mask)
    print(time.perf_counter()-a)

    a = time.perf_counter()
    with ThreadPoolExecutor(6) as ex:
        for _ in range(SAMP_SIZE):
            ex.submit(thread_f_order_mask2rle, rand_mask)
    print(time.perf_counter()-a)