import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# sample data
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Funding Amount (Billion USD)': [10, 15, 20, 25, 30, 35],
    'Number of Startups Funded': [100, 150, 200, 250, 300, 350]
}   
df = pd.DataFrame(data) 
# Set the style of the plot
sns.set_style("whitegrid")  
# Create a line plot for funding amount over the years
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Funding Amount (Billion USD)', data=df, marker='o', label='Funding Amount')
# Create a line plot for the number of startups funded over the years
sns.lineplot(x='Year', y='Number of Startups Funded', data=df, marker='o', label='Number of Startups Funded')
# Set the title and labels  
plt.title('US Startup Funding Trends (2015-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Amount / Number', fontsize=12)
plt.legend()
plt.show()
# other themes
sns.set_style("darkgrid")  
plt.figure(figsize=(10, 6)) 
sns.lineplot(x='Year', y='Funding Amount (Billion USD)', data=df, marker='o', label='Funding Amount')
plt.title('US Startup Funding Trends (2015-2020)', fontsize=16) 
plt.xlabel('Year', fontsize=12)
plt.ylabel('Amount / Number', fontsize=12)  
plt.legend()
plt.show()
sns.set_style("dark")  
plt.figure(figsize=(10, 6)) 
sns.lineplot(x='Year', y='Funding Amount (Billion USD)', data=df, marker='o', label='Funding Amount')
plt.title('US Startup Funding Trends (2015-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12) 
plt.ylabel('Amount / Number', fontsize=12)
plt.legend()
plt.show()
sns.set_style("white")  
plt.figure(figsize=(10, 6)) 
sns.lineplot(x='Year', y='Funding Amount (Billion USD)', data=df, marker='o', label='Funding Amount')
plt.title('US Startup Funding Trends (2015-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Amount / Number', fontsize=12)
plt.legend()
plt.show()
# customize the theme
sns.set_style("whitegrid", {'axes.facecolor': 'lightblue'}) 
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Funding Amount (Billion USD)', data=df, marker='o', label='Funding Amount')
plt.title('US Startup Funding Trends (2015-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Amount / Number', fontsize=12)
plt.legend()    
plt.show()
# load the dataset
data = pd.read_csv('startup_growth_investment_data.csv')
# display the types of the dataset
print(data.dtypes)
print(data.info())
print(data.head(10))
print(data.describe())

sns.set_style("whitegrid")
# kind = 'hist' for histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Growth Rate (%)'], bins=20, kde=True)
plt.title('Distribution of Growth Rates', fontsize=16)
plt.xlabel('Growth Rate (%)', fontsize=12)  
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'bar' for bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Industry', y='Investment Amount (USD)', data=data)
plt.xticks(rotation=45)
plt.title('Investment Amount by Industry', fontsize=16)
plt.xlabel('Industry', fontsize=12)
plt.ylabel('Investment Amount (USD)', fontsize=12)
plt.tight_layout()
plt.show()

#kind = 'scatter' for scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Investment Amount (USD)', y='Valuation (USD)', data=data)
plt.title('Investment Amount vs Valuation', fontsize=16)
plt.xlabel('Investment Amount (USD)', fontsize=12)
plt.ylabel('Valuation (USD)', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'box' for box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Funding Rounds', y='Valuation (USD)', data=data)
plt.title('Valuation by Funding Rounds', fontsize=16)
plt.xlabel('Funding Rounds', fontsize=12)   
plt.ylabel('Valuation (USD)', fontsize=12)
plt.tight_layout()
plt.show()

#kind = 'heatmap' for heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=10)
plt.tight_layout()
plt.show()

#kind = scatter plot with hue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Number of Investors',y='Growth Rate (%)',data=data, hue='Industry')
plt.title('Growth Rate vs Number of Investors by Industry', fontsize=16)
plt.xlabel('Number of Investors', fontsize=12)  
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.legend(title='Industry')
plt.tight_layout()
plt.show()