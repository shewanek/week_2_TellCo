import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
# import pymysql
import time

class SatisfactionAnalysis:
    def __init__(self, df):
        self.clean_df = df


    def assign_enga_exp_scores(self):
        # Engagement columns
        engagement_columns = [
            'Dur. (s)',                    # Session Duration
            'Total UL (Bytes)',             # Total Uplink Traffic
            'Total DL (Bytes)'              # Total Downlink Traffic
        ]
        
        # Experience columns
        experience_columns = [
            'TCP DL Retrans. Vol (Bytes)',  # TCP DL Retransmission
            'Avg RTT DL (ms)',              # Round Trip Time Downlink
            'Avg Bearer TP DL (kbps)'       # Throughput Downlink
        ]
        
        # Handle missing values by imputing with the mean
        imputer = SimpleImputer(strategy='mean')
        engagement_imputed = imputer.fit_transform(self.clean_df[engagement_columns])
        experience_imputed = imputer.fit_transform(self.clean_df[experience_columns])
        
        # Normalize the data
        scaler = StandardScaler()
        scaled_engagement = scaler.fit_transform(engagement_imputed)
        scaled_experience = scaler.fit_transform(experience_imputed)
        
        # KMeans clustering (use K=3 for engagement, K=3 for experience)
        kmeans_engagement = KMeans(n_clusters=3, random_state=0).fit(scaled_engagement)
        kmeans_experience = KMeans(n_clusters=3, random_state=0).fit(scaled_experience)
        
        # Assign engagement and experience scores based on Euclidean distance
        self.clean_df['engagement_score'] = np.linalg.norm(scaled_engagement - kmeans_engagement.cluster_centers_[0], axis=1)
        self.clean_df['experience_score'] = np.linalg.norm(scaled_experience - kmeans_experience.cluster_centers_[0], axis=1)
        
        print("Engagement and experience scores assigned to users.")


    