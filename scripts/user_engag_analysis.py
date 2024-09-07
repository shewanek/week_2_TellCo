import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class User_Engag_Analysis:
    def __init__(self, df):
        self.df = df
        self.agg_df = None
        self.clustered_df = None

    def aggr_user_metrics(self):
        """Aggregate session frequency, session duration, and traffic per user (MSISDN)."""
        self.agg_df = self.df.groupby('MSISDN/Number').agg({
            'Dur. (s)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).rename(columns={
            'Dur. (s)': 'total_duration', 
            'Total DL (Bytes)': 'total_dl', 
            'Total UL (Bytes)': 'total_ul'
        })
        
        self.agg_df['total_traffic'] = self.agg_df['total_dl'] + self.agg_df['total_ul']
        return self.agg_df
