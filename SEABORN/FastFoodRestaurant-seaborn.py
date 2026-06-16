import seaborn as sns  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({'x':np.arange(100), 'y':np.random.rand(100).cumsum()})

# Set the style of the plot
sns.set_style('whitegrid')

# Create a line plot
sns.lineplot(x='x', y='y', data=data)
plt.title('Line Plot with Whitegrid Style')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Set the style to 'darkgrid'
sns.set_style('darkgrid')   
sns.lineplot(x='x', y='y', data=data)
plt.title('Line Plot with Darkgrid Style')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Set the style to 'dark'
sns.set_style('dark')
sns.lineplot(x='x', y='y', data=data)
plt.title('Line Plot with Dark Style')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Set the style to 'white'
sns.set_style('white')  
sns.lineplot(x='x', y='y', data=data)
plt.title('Line Plot with White Style')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Customize the theme
sns.set_style('whitegrid', {'axes.facecolor': 'lightblue'})
sns.lineplot(x='x', y='y', data=data)
plt.title('Line Plot with Customized Theme')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# load the dataset
data = pd.read_csv('FastFoodRestaurants.csv')

# print the dataset
print(data.head())
# print the dataset info
print(data.info())
# print the dataset description
print(data.describe())
# print the dataset types
print(data.dtypes)

sns.set_style('whitegrid')
#kind='hist'
g=sns.displot(x='country', data=data, hue='country', kind='hist')
g.figure.suptitle('sns.displot(x="country", data=data, hue="country", kind="hist")', fontsize=16)
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

#kind='kde'
g=sns.displot(x='latitude', data=data, hue='country', kind='kde')
g.figure.suptitle('sns.displot(x="latitude", data=data, hue="country", kind="kde")', fontsize=16)
plt.xlabel('Latitude')
plt.ylabel('Density')
plt.show()  

#kind='kde'
g=sns.kdeplot(data=data, x='latitude')
g.figure.suptitle('sns.kdeplot(data=data, x="latitude")', fontsize=16)
plt.xlabel('Latitude')
plt.ylabel('Density')
plt.show()

# use seaborn to create a scatter plot of longitude vs latitude
sns.scatterplot(x='longitude', y='latitude', data=data, hue='country')  
plt.title('Scatter plot of longitude vs latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# use seaborn to create a joint plot of longitude vs latitude
g = sns.jointplot(x='longitude', y='latitude', data=data, hue='country')
g.fig.suptitle('sns.jointplot(x="longitude", y="latitude", data=data, hue="country")', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# use seaborn to create a pair plot of the dataset
g = sns.pairplot(data=data, hue='country')
g.fig.suptitle('sns.pairplot(data=data, hue="country")', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# use seaborn to create a box plot of latitude by country
g=sns.catplot(x='country', y='latitude', data=data, kind='box')
g.fig.suptitle('sns.catplot(x="country", y="latitude", data=data, kind="box")', fontsize=16)
plt.xlabel('Country')
plt.ylabel('Latitude')
plt.show()

# use seaborn to create a histogram of latitude by country
g=sns.histplot(x='latitude', data=data, hue='country', kde=True)
g.figure.suptitle('sns.histplot(x="latitude", data=data, hue="country", kde=True)', fontsize=16)
plt.xlabel('Latitude')
plt.ylabel('Count')
plt.show()