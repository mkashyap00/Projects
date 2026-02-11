import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class BroadbandDeploymentAnalyzer:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        print("Initial data shape:", self.df.shape)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data"""
        # Define numeric columns we'll use
        self.numeric_columns = [
            'total_population', 'housing_units', 'median_household_income',
            'population_25_plus', 'median_home_value'  # Removed total_families as it was NaN
        ]
        
        # Convert to numeric and handle invalid values
        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].mask(self.df[col] < 0)
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        print("\nSample of cleaned numeric data:")
        print(self.df[self.numeric_columns].head())
        
    def calculate_priority_score(self):
        """Calculate priority scores using refined weights"""
        # Adjusted weights without total_families
        weights = {
            'total_population': 0.30,
            'housing_units': 0.25,
            'median_household_income': 0.20,
            'population_25_plus': 0.15,
            'median_home_value': 0.10
        }
        
        print("\nCalculating priority scores with weights:", weights)
        
        # Normalize features
        scaler = StandardScaler()
        features = list(weights.keys())
        normalized_data = scaler.fit_transform(self.df[features])
        normalized_df = pd.DataFrame(normalized_data, columns=features, index=self.df.index)
        
        # Calculate weighted score
        self.df['priority_score'] = 0
        for col, weight in weights.items():
            self.df['priority_score'] += normalized_df[col] * weight
        
        print("\nPriority score statistics:")
        print(self.df['priority_score'].describe())
        
        return self.create_summary()
    
    def create_summary(self):
        """Create summary statistics"""
        summary = {
            'bidder_stats': self.df.groupby('bidder').agg({
                'priority_score': ['mean', 'count'],
                'total_population': 'sum',
                'housing_units': 'sum',
                'median_household_income': 'mean'
            }).round(2),
            
            'top_priority_areas': self.df.nlargest(10, 'priority_score')[
                ['bidder', 'state', 'priority_score', 'total_population', 'housing_units', 'median_household_income']
            ]
        }
        return summary
    
    def create_visualizations(self):
        """Create visualizations"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Priority Score Distribution
        sns.histplot(data=self.df, x='priority_score', bins=30, ax=axes[0,0])
        axes[0,0].set_title('Distribution of Priority Scores')
        
        # 2. Top 10 Bidders
        top_bidders = self.df.groupby('bidder')['priority_score'].mean().nlargest(10)
        axes[0,1].barh(y=range(len(top_bidders)), width=top_bidders.values)
        axes[0,1].set_yticks(range(len(top_bidders)))
        axes[0,1].set_yticklabels(top_bidders.index, fontsize=8)
        axes[0,1].set_title('Top 10 Bidders by Average Priority Score')
        
        # 3. Population vs Priority Score
        axes[1,0].scatter(self.df['total_population'], self.df['priority_score'], alpha=0.5)
        axes[1,0].set_xlabel('Total Population')
        axes[1,0].set_ylabel('Priority Score')
        axes[1,0].set_title('Population vs Priority Score')
        
        # 4. Income vs Priority Score
        axes[1,1].scatter(self.df['median_household_income'], self.df['priority_score'], alpha=0.5)
        axes[1,1].set_xlabel('Median Household Income')
        axes[1,1].set_ylabel('Priority Score')
        axes[1,1].set_title('Income vs Priority Score')
        
        plt.tight_layout()
        plt.savefig('priority_analysis.png')
        print("\nSaved priority analysis plot to 'priority_analysis.png'")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[self.numeric_columns + ['priority_score']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        print("Saved correlation matrix to 'correlation_matrix.png'")

def main():
    input_file = "census_data_results_sample_20241201_161107.csv"
    print(f"Analyzing data from: {input_file}")
    
    analyzer = BroadbandDeploymentAnalyzer(input_file)
    results = analyzer.calculate_priority_score()
    
    if results:
        print("\nTop Priority Areas:")
        print(results['top_priority_areas'])
        
        print("\nBidder Statistics:")
        print(results['bidder_stats'])
        
        analyzer.create_visualizations()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()