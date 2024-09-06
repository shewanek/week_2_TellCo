import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TellCoEDA:
    def __init__(self, df):
        self.df = df

    # Step 2: Data wangleing 
    def wrangle(self):
        # Drop the features with high null values
        self.df.drop(
            columns=[
                "TCP DL Retrans. Vol (Bytes)", 
                "TCP UL Retrans. Vol (Bytes)", 
                "HTTP DL (Bytes)", 
                "HTTP UL (Bytes)",
                "Nb of sec with 125000B < Vol DL", 
                "Nb of sec with 1250B < Vol UL < 6250B", 
                "Nb of sec with 31250B < Vol DL < 125000B", 
                "Nb of sec with 37500B < Vol UL", 
                "Nb of sec with 6250B < Vol DL < 31250B", 
                "Nb of sec with 6250B < Vol UL < 37500B"
            ], 
            inplace=True
        )
        # self.df.fillna(self.df.mean(), inplace=True)
        # self.df.drop_duplicates(inplace=True)
        return self.df

    # Step 3: Descriptive Statistics
    def descriptive_stats(self):
        return self.df.describe()

    # Step 4: Data Transformation
    def transform_data(self):
        self.df['Total Data Volume'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']
        self.df['Duration Decile'] = pd.qcut(self.df['Dur. (s)'], 10, labels=False)

    # Step 5: Univariate Analysis
    def univariate_analysis(self):
        mean_duration = self.df['Dur. (s)'].mean()
        median_duration = self.df['Dur. (s)'].median()
        variance_duration = self.df['Dur. (s)'].var()

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['Dur. (s)'], bins=20, kde=True)
        plt.title('Distribution of Session Duration')
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df['Dur. (s)'])
        plt.title('Boxplot of Session Duration')
        plt.show()

    # Step 6: Bivariate Analysis
    def bivariate_analysis(self):
        applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)',
                        'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        for app in applications:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=self.df[app], y=self.df['Total Data Volume'])
            plt.title(f'{app} vs Total Data Volume')
            plt.show()

    # Step 7: Correlation Analysis
    def correlation_analysis(self):
        applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)',
                        'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        app_data = self.df[applications]
        correlation_matrix = app_data.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Application Data Usage')
        plt.show()

    # Step 8: Dimensionality Reduction using PCA
    def perform_pca(self):
        applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)',
                        'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[applications])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        plt.figure(figsize=(10, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.df['Duration Decile'], cmap='viridis', s=50)
        plt.title('PCA Result: Data Usage')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.show()

    # Step 9: Insights for Marketing Team
    def marketing_insights(self):
        top_10_handsets = self.df['Handset Type'].value_counts().head(10)
        top_3_manufacturers = self.df['Handset Manufacturer'].value_counts().head(3)

        top_handsets_per_manufacturer = {}
        for manufacturer in top_3_manufacturers.index:
            top_handsets_per_manufacturer[manufacturer] = self.df[self.df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)

        return top_10_handsets, top_3_manufacturers, top_handsets_per_manufacturer
