import pandas as pd

# Load the dataset
data = pd.read_csv('realestate-USA.csv')

# Display information about the dataset
data.info()    
print(data.info())

# return the count of unique values in the 'status' column
unique_status_count = data['status'].nunique()
print(unique_status_count)

# shows the frequency of each unique value in the 'status' column
status_counts = data['status'].value_counts()   
print(status_counts)

# duplicate rows in the dataset
duplicate_rows = data.duplicated()
print(duplicate_rows)

# Display the first few rows of the dataset
print(data.head())
# Display summary statistics of the dataset
print(data.describe())
# display the last three rows of the dataset
print(data.tail(3))
# dispaly the first three rows of the dataset
print(data.head(3))
# access the name column of the dataset
print(data['status']) 
# selecting the single row using the .loc method
print(data.loc[:, 'status'])
# selecting multiple columns using the .loc method
print(data.loc[:, ['status', 'price']])
# selecting the single row using the .iloc method
print(data.iloc[:, 1])
# selecting multiple columns using the .iloc method
print(data.iloc[:, [1, 2]])
# selecting the slice of rows using the .loc method
print(data.loc[0:3, ['status', 'price']])
# combining the .loc and .iloc methods to select specific rows and columns
print(data.loc[0:3].iloc[:, [1, 2]])
# combined row and column selection using the .loc method
print(data.loc[0:3, 'status':'price'])
# summary of statistics of data frame using the describe() method
print(data.describe())
# counting the rows and columns in data frame using the shape attribute
print(data.shape)
# access the name column of the dataset
print(data['city']) 

# access multiple columns of the dataset
print(data[['city', 'state']]) 
# delete row with index 1
data.drop(1, inplace=True)
# display the first few rows of the dataset after deleting the row  
print(data.head())

# delete column with name 'status'
data.drop('status', axis=1, inplace=True)
# display the first few rows of the dataset after deleting the column
print(data.head())

# rename the column 'price' to 'cost'
data.rename(columns={'price': 'cost'}, inplace=True)
# display the first few rows of the dataset after renaming the column
print(data.head())

# filter the dataset to include only properties in California
california_properties = data[data['state'] == 'California']
# display the first few rows of the filtered dataset
print(california_properties.head())

# filter the dataset to include only properties with a cost greater than $500,000
expensive_properties = data[data['cost'] > 500000]
# display the first few rows of the filtered dataset
print(expensive_properties.head())  

# select the row where price is maximum
max_price_row = data[data['cost'] == data['cost'].max()]    
# sort the dataset by cost in descending order
sorted_data = data.sort_values(by='cost', ascending=False)
# display the first few rows of the sorted dataset
print(sorted_data.head())

# group the dataset by state and calculate the average cost for each state
average_cost_by_state = data.groupby('state')['cost'].mean()
print(average_cost_by_state)

# sort dataframe of pandas by multiple columns
sorted_data_multi = data.sort_values(by=['state', 'cost'], ascending=[True, False])
# display the first few rows of the sorted dataset
print(sorted_data_multi.head())

#calculate the sum of price for each state
total_cost_by_state = data.groupby('state')['cost'].sum()
print(total_cost_by_state)
grouped = data.groupby('state')
for price, group in grouped:
    print(price)
    print(group['city'])

# use dropna() method to remove rows with missing values
cleaned_data = data.dropna()
# display the first few rows of the cleaned dataset
print(cleaned_data.head())

# use fillna() method to fill missing values with the mean cost
data['cost'].fillna(data['cost'].mean(), inplace=True)
# display the first few rows of the dataset after filling missing values
print(data.head())

# create a new column 'price_category' based on the cost of the properties
data['price_category'] = pd.cut(data['cost'], bins=[0, 250000, 500000, 1000000, float('inf')], labels=['Low', 'Medium', 'High', 'Luxury'])
# display the first few rows of the dataset after creating the new column   
print(data.head())

# creating a pandas .array from a list of values
price_array = pd.array([250000, 500000, 750000, 1000000])
print(price_array)

 # creating a pandas Series from a list of values
price_series = pd.Series([250000, 500000, 750000, 1000000])
print(price_series)
