import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class User_Engag_Analysis:
    def __init__(self, df):
        """Initialize the class with a dataframe."""
        self.df = df
        self.agg_df = None
        self.clustered_df = None

    def aggr_user_metrics(self):
        """
        Aggregate session metrics (session duration and traffic) per user (MSISDN/Number).
        Report the top 10 customers per engagement metric.
        """
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
        
        # Top 10 customers by engagement metrics
        print("Top 10 customers by session duration:\n", self.agg_df.nlargest(10, 'total_duration'))
        print("Top 10 customers by download traffic:\n", self.agg_df.nlargest(10, 'total_dl'))
        print("Top 10 customers by upload traffic:\n", self.agg_df.nlargest(10, 'total_ul'))
        print("Top 10 customers by total traffic:\n", self.agg_df.nlargest(10, 'total_traffic'))
        
        return self.agg_df

    def normalize_metrics(self):
        """
        Normalize the engagement metrics (duration, download, upload, and total traffic).
        """
        scaler = MinMaxScaler()
        self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']] = scaler.fit_transform(
            self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
        return self.agg_df

    def run_kmeans(self, k=3):
        """
        Apply K-means clustering to the normalized metrics and visualize the clusters using pairplot.
        """
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.agg_df['cluster'] = kmeans.fit_predict(self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
        self.clustered_df = self.agg_df
        
        # Pairplot visualization for cluster separation
        sns.pairplot(self.clustered_df, hue='cluster', diag_kind='kde', 
                     vars=['total_duration', 'total_dl', 'total_ul', 'total_traffic'], palette='Set2', corner=True)
        plt.suptitle('K-means Clustering Results', y=1.02)
        plt.show()

        return self.clustered_df

    def plot_elbow_method(self):
        """
        Use the elbow method to find the optimal number of clusters (k).
        """
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
            distortions.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(K, distortions, 'bo-', color='red')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def compute_cluster_stats(self):
        """
        Compute and display statistics for each cluster (min, max, avg, total) for non-normalized metrics.
        """
        non_normalized_df = self.df.groupby('MSISDN/Number').agg({
            'Dur. (s)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).rename(columns={
            'Dur. (s)': 'total_duration', 
            'Total DL (Bytes)': 'total_dl', 
            'Total UL (Bytes)': 'total_ul'
        })
        non_normalized_df['total_traffic'] = non_normalized_df['total_dl'] + non_normalized_df['total_ul']
        
        non_normalized_df['cluster'] = self.clustered_df['cluster']
        
        cluster_stats = non_normalized_df.groupby('cluster').agg({
            'total_duration': ['min', 'max', 'mean', 'sum'],
            'total_dl': ['min', 'max', 'mean', 'sum'],
            'total_ul': ['min', 'max', 'mean', 'sum'],
            'total_traffic': ['min', 'max', 'mean', 'sum']
        })
        
        print("Cluster Statistics:\n", cluster_stats)
        return cluster_stats

    def plot_top_apps(self):
        """
        Plot the top 3 most used applications by total data usage (download traffic).
        """
        top_apps = self.df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 
                            'Netflix DL (Bytes)', 'Gaming DL (Bytes)']].sum().nlargest(3)
        
        top_apps.plot(kind='bar', figsize=(10, 6), color='skyblue')
        plt.title('Top 3 Applications by Data Usage')
        plt.ylabel('Total Download (Bytes)')
        plt.show()

    def aggregate_top_users_per_app(self):
        """
        Aggregate total traffic per application and report top 10 most engaged users.
        """
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 
                       'Netflix DL (Bytes)', 'Gaming DL (Bytes)']
        
        top_users_per_app = self.df.groupby('MSISDN/Number')[app_columns].sum().nlargest(10, app_columns)
        print("Top 10 most engaged users per application:\n", top_users_per_app)
        return top_users_per_app
