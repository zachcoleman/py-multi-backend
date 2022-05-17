import numpy as np
from typing import Dict
from itertools import groupby
from pycocotools import mask as cocomask

def mask2rle(mask: np.ndarray) -> Dict:
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


def rle2mask(seg_rle: Dict) -> np.ndarray:
    if type(seg_rle["counts"]) is str:
        seg_rle["counts"] = seg_rle["counts"].encode("utf-8")
    seg_mask = np.array(cocomask.decode(seg_rle), dtype=np.uint8)
    return np.array(seg_mask)