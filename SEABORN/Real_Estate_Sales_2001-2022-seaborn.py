import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt   

# sample data
data = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})

# Set the style of the plot
sns.set_style("whitegrid") 

# Create a line plot for x and y
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data, marker='o')

# Set the title and labels
plt.title('Line Plot Example', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.show()

# other themes  
sns.set_style("darkgrid") 
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data, marker='o')
plt.title('Line Plot Example', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.show()

sns.set_style("dark") 
plt.figure(figsize=(10, 6)) 
sns.lineplot(x='x', y='y', data=data, marker='o')
plt.title('Line Plot Example', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.show()

sns.set_style("white")
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data, marker='o')
plt.title('Line Plot Example', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.show()

# customize the theme
sns.set_style("whitegrid", {'axes.facecolor': 'lightblue'})
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data, marker='o')
plt.title('Line Plot Example', fontsize=16)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.show()

# load the dataset
data = pd.read_csv('Real_Estate_Sales_2001-2022_GL-Short.csv')
# display basic information about the dataset  
print(data.dtypes)
print(data.info())
print(data.head(10))
print(data.describe())

sns.set_style("whitegrid")
# kind = 'hist' for histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales Ratio'], bins=20, kde=True)
plt.title('Distribution of Sales Ratio', fontsize=16)
plt.xlabel('Sales Ratio', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'bar' for bar plot 
plt.figure(figsize=(10, 6))
sns.barplot(x='Property Type', y='Sale Amount', data=data)
plt.title('Average Sale Amount by Property Type', fontsize=16)
plt.xlabel('Property Type', fontsize=12)
plt.ylabel('Average Sale Amount', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'scatter' for scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Assessed Value', y='Sale Amount', data=data)
plt.title('Assessed Value vs Sale Amount', fontsize=16)
plt.xlabel('Assessed Value', fontsize=12)
plt.ylabel('Sale Amount', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'box' for box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Residential Type', y='Sale Amount', data=data)
plt.title('Sale Amount by Residential Type', fontsize=16)
plt.xlabel('Residential Type', fontsize=12)
plt.ylabel('Sale Amount', fontsize=12)
plt.tight_layout()
plt.show()

# kind = 'heatmap' for heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=10)
plt.tight_layout()
plt.show()

# kind = scatter plot with hue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Assessed Value', y='Sale Amount', hue='Property Type', data=data)
plt.title('Assessed Value vs Sale Amount by Property Type', fontsize=16)        
plt.xlabel('Assessed Value', fontsize=12)
plt.ylabel('Sale Amount', fontsize=12)
plt.legend(title='Property Type')
plt.tight_layout()
plt.show()