# -*- coding: utf-8 -*-
"""

@author: Carlos
"""

import pandas as pd

# Criteo dataset to 4000 samples

csv1 = pd.read_csv("../data/criteo/train.csv")
csv2 = pd.read_csv("../data/criteo/test.csv")

csv = pd.concat([csv1, csv2], axis=0)
csv.to_csv("../data/criteo/criteo_sample.csv")
