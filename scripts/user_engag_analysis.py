import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

    def normalize_metrics(self):
        """Normalize each engagement metric."""
        scaler = MinMaxScaler()
        self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']] = scaler.fit_transform(
            self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
        return self.agg_df

    # def run_kmeans(self, k=3):
    #     """Apply k-means clustering to the normalized metrics."""
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     self.agg_df['cluster'] = kmeans.fit_predict(self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
    #     self.clustered_df = self.agg_df
    #     return self.clustered_df
    
   

    def run_kmeans(self, k=3):
        """Apply k-means clustering to the normalized metrics and visualize the clusters."""
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.agg_df['cluster'] = kmeans.fit_predict(self.agg_df[['total_duration', 'total_dl', 'total_ul', 'total_traffic']])
        self.clustered_df = self.agg_df
        
        # Visualization using pair plot to see cluster separation
        sns.pairplot(self.clustered_df, hue='cluster', diag_kind='kde', 
                    vars=['total_duration', 'total_dl', 'total_ul', 'total_traffic'], palette='Set2', corner=True)
        plt.suptitle('K-means Clustering Results', y=1.02)
        plt.show()

        return self.clustered_df


    def plot_elbow_method(self):
        """Use the elbow method to find the optimal value of k."""
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

    def plot_top_apps(self):
        """Plot the top 3 most used applications (Total Traffic)."""
        top_apps = self.df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 
                            'Netflix DL (Bytes)', 'Gaming DL (Bytes)']].sum().nlargest(3)
        
        top_apps.plot(kind='bar', figsize=(10, 6), color='skyblue')
        plt.title('Top 3 Applications by Data Usage')
        plt.ylabel('Total Download (Bytes)')
        plt.show()

