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

    def wrangle(self):
        """
        Treat missing values and outliers by replacing them with the mean or mode, 
        and handle outliers by capping values at the 10th and 90th percentiles.
        """
        # Replace missing numeric values with the mean
        numeric_cols = ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 
                        'Avg RTT UL (ms)', 'Avg Bearer TP UL (kbps)']
        for col in numeric_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Replace missing categorical values with the mode
        self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0], inplace=True)

        # Handle outliers by capping at the 10th and 90th percentiles
        for col in numeric_cols:
            self.df[col] = np.where(self.df[col] > self.df[col].quantile(0.90), self.df[col].quantile(0.90), self.df[col])
            self.df[col] = np.where(self.df[col] < self.df[col].quantile(0.10), self.df[col].quantile(0.10), self.df[col])

        self.clean_df = self.df
        return self.clean_df


    def aggr_user_metrics(self):
        """
        Aggregate the following per customer (MSISDN/Number):
        - Average TCP retransmission
        - Average RTT
        - Handset type
        - Average throughput
        """
        self.agg_df = self.clean_df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode()[0]  # Most frequent Handset type
        }).rename(columns={
            'TCP DL Retrans. Vol (Bytes)': 'avg_tcp_retrans',
            'Avg RTT DL (ms)': 'avg_rtt',
            'Avg Bearer TP DL (kbps)': 'avg_throughput'
        })
        return self.agg_df

    def compute_top_bottom_frequent(self):
        """
        Compute and list 10 of the top, bottom, and most frequent values for:
        - TCP retransmission
        - RTT
        - Throughput
        """
        top_tcp = self.df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
        bottom_tcp = self.df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
        frequent_tcp = self.df['TCP DL Retrans. Vol (Bytes)'].value_counts().nlargest(10)
        
        top_rtt = self.df['Avg RTT DL (ms)'].nlargest(10)
        bottom_rtt = self.df['Avg RTT DL (ms)'].nsmallest(10)
        frequent_rtt = self.df['Avg RTT DL (ms)'].value_counts().nlargest(10)
        
        top_throughput = self.df['Avg Bearer TP DL (kbps)'].nlargest(10)
        bottom_throughput = self.df['Avg Bearer TP DL (kbps)'].nsmallest(10)
        frequent_throughput = self.df['Avg Bearer TP DL (kbps)'].value_counts().nlargest(10)
        
        return {
            'top_tcp': top_tcp, 'bottom_tcp': bottom_tcp, 'frequent_tcp': frequent_tcp,
            'top_rtt': top_rtt, 'bottom_rtt': bottom_rtt, 'frequent_rtt': frequent_rtt,
            'top_throughput': top_throughput, 'bottom_throughput': bottom_throughput, 'frequent_throughput': frequent_throughput
        }

    def report_distributions(self):
        """
        Compute and report the distribution of:
        - Average throughput per handset type
        - Average TCP retransmission per handset type
        """
        throughput_distribution = self.clean_df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False)
        tcp_retrans_distribution = self.clean_df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)
        
        # Plot the distribution of average throughput per handset type
        plt.figure(figsize=(10, 6))
        throughput_distribution.plot(kind='bar', color='skyblue')
        plt.title('Average Throughput per Handset Type')
        plt.ylabel('Average Throughput (kbps)')
        plt.xlabel('Handset Type')
        plt.show()
        
        # Plot the distribution of average TCP retransmission per handset type
        plt.figure(figsize=(10, 6))
        tcp_retrans_distribution.plot(kind='bar', color='salmon')
        plt.title('Average TCP Retransmission per Handset Type')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.xlabel('Handset Type')
        plt.show()

        return throughput_distribution, tcp_retrans_distribution

    def run_kmeans_clustering(self, k=3):
        """
        Perform K-Means clustering (k=3) to segment users based on their experience.
        """
        scaler = MinMaxScaler()
        self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']] = scaler.fit_transform(
            self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.agg_df['cluster'] = kmeans.fit_predict(self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']])
        
        self.clustered_df = self.agg_df
        
        # Visualize the clusters
        sns.pairplot(self.clustered_df, hue='cluster', diag_kind='kde', 
                     vars=['avg_tcp_retrans', 'avg_rtt', 'avg_throughput'], palette='Set2', corner=True)
        plt.suptitle('K-means Clustering Results - User Experience Clusters', y=1.02)
        plt.show()

        return self.clustered_df

    def describe_clusters(self):
        """
        Provide a description of each cluster based on user experience metrics.
        """
        cluster_description = self.clustered_df.groupby('cluster').agg({
            'avg_tcp_retrans': ['min', 'max', 'mean'],
            'avg_rtt': ['min', 'max', 'mean'],
            'avg_throughput': ['min', 'max', 'mean'],
            'Handset Type': lambda x: x.mode()[0]  # Most common handset type per cluster
        })
        print("Cluster Descriptions:\n", cluster_description)
        return cluster_description

