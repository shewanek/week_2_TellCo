import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class Exp_Analytics:
    def __init__(self, df):
        """Initialize with the dataset."""
        self.df = df
        self.agg_df = None
        self.clustered_df = None

