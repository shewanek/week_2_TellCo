import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TellCoEDA:
    def __init__(self, df):
        self.df = df

    # Step 1: Data wangleing 
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
        
        # Drop low and high cardinality categorical variable
        self.df.drop(columns=["Last Location Name"], inplace=True)

        # Calculate Q1, Q3, and IQR for numeric columns
        Q1 = self.df[['Dur. (s)', 'Total DL (Bytes)', 'Total UL (Bytes)']].quantile(0.10)  # Changed to 0.10 for Q1
        Q3 = self.df[['Dur. (s)', 'Total DL (Bytes)', 'Total UL (Bytes)']].quantile(0.90)  # Changed to 0.90 for Q3
        IQR = Q3 - Q1

        # Filter for outliers in numeric columns
        outlier_condition = ((self.df[['Dur. (s)', 'Total DL (Bytes)', 'Total UL (Bytes)']] < (Q1 - 1.5 * IQR)) |
                            (self.df[['Dur. (s)', 'Total DL (Bytes)', 'Total UL (Bytes)']] > (Q3 + 1.5 * IQR))).any(axis=1)

        # Remove outliers
        self.df = self.df[~outlier_condition]

        # Fill missing values with the mean, but only for numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        return self.df




    def transform_data(self):
        # Create a new column for Total Data Volume
        self.df['Total Data Volume'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']
        
        # Create decile groups for Duration, handling duplicate bin edges
        self.df['Duration Decile'] = pd.qcut(self.df['Dur. (s)'], 10, labels=False, duplicates='drop')
        
        return self.df


    # Step 4: Univariate Analysis
    def univariate_analysis(self):
        mean_duration = self.df['Dur. (s)'].mean()
        median_duration = self.df['Dur. (s)'].median()
        variance_duration = self.df['Dur. (s)'].var()
         # Print statistics
        print(f"Mean Duration: {mean_duration}")
        print(f"Median Duration: {median_duration}")
        print(f"Variance Duration: {variance_duration}")

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['Dur. (s)'], bins=20, kde=True)
        plt.title('Distribution of Session Duration')
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df['Dur. (s)'])
        plt.title('Boxplot of Session Duration')
        plt.show()
