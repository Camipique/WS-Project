# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

import numpy as np
import pandas as pd

# Avazu dataset to 4000 samples

n_rows = 4577464
skip = np.delete(np.arange(n_rows), np.arange(0, n_rows, 2290))

csv = pd.read_csv("../data/avazu/test.csv", skiprows = skip)
csv.to_csv("../data/avazu/avazu_sample.csv")

