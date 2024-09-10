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


    def calculate_satisfaction(self):
        # Calculate satisfaction score as the average of engagement and experience scores
        self.clean_df['satisfaction_score'] = (self.clean_df['engagement_score'] + self.clean_df['experience_score']) / 2
        
        # Get top 10 satisfied customers
        top_10_satisfied = self.clean_df[['MSISDN/Number', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10)
        
        print("Top 10 satisfied customers:")
        print(top_10_satisfied)
        
        return top_10_satisfied

    def train_regression_model(self):
        # Features: engagement and experience scores
        X = self.clean_df[['engagement_score', 'experience_score']]
        y = self.clean_df['satisfaction_score']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model MSE: {mse}")
        print(f"Model R2 score: {r2}")
        
        return model

    def kmeans_on_scores(self):
        # Engagement and experience scores
        score_columns = ['engagement_score', 'experience_score']
        scaled_scores = StandardScaler().fit_transform(self.clean_df[score_columns])
        
        # KMeans clustering (K=2)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_scores)
        
        # Assign clusters
        self.clean_df['satisfaction_cluster'] = kmeans.labels_
        
        print("K-Means clustering on engagement and experience scores done.")

    def aggregate_scores_per_cluster(self):
        # Aggregate satisfaction and experience scores per cluster
        cluster_aggregation = self.clean_df.groupby('satisfaction_cluster').agg({
            'satisfaction_score': ['mean', 'std'],
            'experience_score': ['mean', 'std']
        }).reset_index()
        
        print("Satisfaction and experience scores per cluster:")
        print(cluster_aggregation)
        
        return cluster_aggregation


    