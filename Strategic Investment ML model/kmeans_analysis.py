import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class BroadbandClusterAnalyzer:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        print("Initial data shape:", self.df.shape)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data"""
        # Define numeric columns we'll use
        self.numeric_columns = [
            'total_population', 'housing_units', 'median_household_income',
            'population_25_plus', 'median_home_value'
        ]
        
        # Convert to numeric and handle invalid values
        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].mask(self.df[col] < 0)
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        print("\nSample of cleaned numeric data:")
        print(self.df[self.numeric_columns].head())

    def perform_kmeans_analysis(self):
        """Perform k-means clustering analysis"""
        # Select features for clustering
        cluster_features = self.numeric_columns
        
        # Normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[cluster_features])
        
        # Determine optimal number of clusters using elbow method
        print("\nCalculating optimal number of clusters...")
        inertias = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.savefig('elbow_curve.png')
        plt.close()
        print("Saved elbow curve plot as 'elbow_curve.png'")
        
        # Perform clustering with k=4
        print("\nPerforming clustering with k=4...")
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_stats = self.df.groupby('cluster').agg({
            'total_population': 'mean',
            'housing_units': 'mean',
            'median_household_income': 'mean',
            'population_25_plus': 'mean',
            'median_home_value': 'mean',
            'bidder': 'count'
        }).round(2)
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        # Visualize clusters
        self.create_cluster_visualizations(cluster_features)
        
        return cluster_stats
    
    def create_cluster_visualizations(self, features):
        """Create visualizations for cluster analysis"""
        # Population vs Income colored by cluster
        plt.figure(figsize=(12, 8))
        plt.scatter(self.df['total_population'], 
                   self.df['median_household_income'],
                   c=self.df['cluster'],
                   cmap='viridis',
                   alpha=0.6)
        plt.xlabel('Total Population')
        plt.ylabel('Median Household Income')
        plt.title('Clusters by Population and Income')
        plt.colorbar(label='Cluster')
        plt.savefig('cluster_analysis_pop_income.png')
        plt.close()
        print("Saved population vs income cluster plot as 'cluster_analysis_pop_income.png'")
        
        # Population vs Housing Units colored by cluster
        plt.figure(figsize=(12, 8))
        plt.scatter(self.df['total_population'], 
                   self.df['housing_units'],
                   c=self.df['cluster'],
                   cmap='viridis',
                   alpha=0.6)
        plt.xlabel('Total Population')
        plt.ylabel('Housing Units')
        plt.title('Clusters by Population and Housing Units')
        plt.colorbar(label='Cluster')
        plt.savefig('cluster_analysis_pop_housing.png')
        plt.close()
        print("Saved population vs housing cluster plot as 'cluster_analysis_pop_housing.png'")
        
        # Analyze bidder distribution across clusters
        bidder_clusters = pd.crosstab(self.df['bidder'], self.df['cluster'])
        plt.figure(figsize=(15, 8))
        bidder_clusters.plot(kind='bar', stacked=True)
        plt.title('Bidder Distribution Across Clusters')
        plt.xlabel('Bidder')
        plt.ylabel('Number of Block Groups')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig('bidder_cluster_distribution.png')
        plt.close()
        print("Saved bidder distribution plot as 'bidder_cluster_distribution.png'")

def main():
    # Specify input file
    input_file = "census_data_results_sample_20241201_161107.csv"  # Update with your file name
    print(f"Analyzing data from: {input_file}")
    
    # Initialize analyzer
    analyzer = BroadbandClusterAnalyzer(input_file)
    
    # Perform clustering analysis
    cluster_stats = analyzer.perform_kmeans_analysis()
    
    print("\nAnalysis complete! Check the generated visualization files.")
    
    return analyzer, cluster_stats

if __name__ == "__main__":
    analyzer, cluster_stats = main()