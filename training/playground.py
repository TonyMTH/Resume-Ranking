from collections import Counter

import numpy as np

corpus = [
    'This is the first document'.lower().split(),
    'This document is the second document .'.lower().split(),
    'And this is the third one.'.lower().split(),
    'Is this the first document?'.lower().split(),
]
query = "this is the third one".lower().split()

print(np.array([[2, 3], [4, 5]]) / np.array([[2, 3], [2, 3]]))
