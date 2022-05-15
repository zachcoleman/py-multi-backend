import numpy as np
from typing import Dict
from itertools import groupby

def py_mask_to_rle(mask: np.ndarray) -> Dict:
    counts = []
    for idx, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if idx == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return {"counts": counts, "size": list(mask.shape)}

# mirrors Rust implementation 
# def py_mask_to_rle(mask: np.ndarray):
#     counts = []
#     tgt = False
#     count = 0

#     for curr in mask.ravel(order="F"):
#         if curr == tgt:
#             count += 1
#         else:
#             counts.append(count)
#             tgt = curr
#             count = 1
#     counts.append(count)

#     return {"counts": counts, "size": list(mask.shape)}