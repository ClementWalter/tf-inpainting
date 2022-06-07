import numpy as np
import tensorflow as tf
import os
import pandas as pd

import re

#%% Read directories
dataset = pd.DataFrame({"filename": os.listdir("data/1000")}).filename.str.extract(
    r"^(?P<geometry>\w+)_U\((?P<y>\d+\.\d+),(?P<x>\d+\.\d+)\)_dpi\((?P<height>\d+),(?P<width>\d+)\).npz"
)


len(dataset.geometry.unique())
dataset.geometry.value_counts()
