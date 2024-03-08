import math
import time
import os
from joblib import Parallel, delayed

import pandas as pd
import colorthief

def factorial(x):
    result = [math.factorial(y) for y in range(x)]
    return result

t1 = time.time()
#results = [math.factorial(x) for x in range(10000)]
results = Parallel(n_jobs=-1)(delayed(factorial)(10000))
results
t2 = time.time()

print(t2-t1)


def factorial(x):
    result = [math.factorial(y) for y in range(x)]
    return result

t1 = time.time()
#results = [math.factorial(x) for x in range(10000)]
results = Parallel(n_jobs=-1)(delayed(math.factorial)(y) for y in range(1000))
results
t2 = time.time()

print(t2-t1)