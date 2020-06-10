# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

from datetime import datetime

import pandas as pd

start_time = datetime.now()

print("Reading data...\n")

data_criteo = pd.read_csv('./data/criteo/criteo_sample.csv')
data_avazu = pd.read_csv('./data/avazu/avazu_sample.csv')





print("Program ended in {time}".format(time = datetime.now() - start_time))