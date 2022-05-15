# cython: language_level=3

import numpy as np
from typing import Dict
from itertools import groupby

# implementation mirrors Rust impl.
def cy_mask_to_rle(mask: np.ndarray):
    cdef list counts = []
    cdef bint curr = False
    cdef bint tgt = False
    cdef int count = 0

    for curr in mask.ravel(order="F"):
        if curr == tgt:
            count += 1
        else:
            counts.append(count)
            tgt = curr
            count = 1
    counts.append(count)

    return {"counts": counts, "size": list(mask.shape)}

# based on itertools.groupby implementation
# def cy_mask_to_rle(mask: np.ndarray):
#     mask_iter = iter(mask.ravel(order="F"))
#     cdef list counts = []
#     cdef bint curr = False
#     cdef bint tgt = False
#     cdef bint isbreak = False
#     cdef int idx = 0
#     cdef int count = 0

#     while True:
#         count = 0
#         tgt = curr
#         while curr == tgt:
#             count += 1
#             try:
#                 curr = next(mask_iter)
#             except StopIteration:
#                 isbreak = True
#                 break
        
#         if len(counts) == 0:
#             if tgt:
#                 counts.append(0)
#             else:
#                 counts.append(count-1)
#         else:
#             counts.append(count)
        
#         if isbreak:
#             break

#     return {"counts": counts, "size": list(mask.shape)}