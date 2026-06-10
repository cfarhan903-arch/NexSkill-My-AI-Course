import pandas as pd 
#load the dataset
data = pd.read_csv('fastfoodrestaurants.csv')
#display the first 5 rows of the dataset
print(data.head())
#display the last 5 rows of the dataset
print(data.tail())
#display the number of rows and columns in the dataset
print(data.shape)
#display the column names of the dataset
print(data.columns)
#display the data types of each column in the dataset
print(data.dtypes)
#display the number of non-null values in each column of the dataset
print(data.count())
#display the summary statistics of the dataset
print(data.describe())
#display the information about the dataset
print(data.info())
#count the number of unique values in the 'name' column
unique_names = data['name'].nunique()
print(unique_names)
#count the frequency of each unique value in the 'address' column
address_freq = data['address'].value_counts()
print(address_freq)
#check for duplicate rows in the dataset
duplicate_rows = data.duplicated()
# check the length of the dataset before removing duplicates
print(len(data))
# remove duplicate rows from the dataset
data.drop_duplicates(inplace=True)
# check the length of the dataset after removing duplicates
print(len(data))
#display the first 5 rows of the dataset after removing duplicates
print(data.head())
#display the information about dataset
print(data.info())
#check for missing values in the dataset
missing_values = data.isnull().sum()
print(missing_values)
#fill missing values in the 'name' column with 'Unknown'
data['name'].fillna('Unknown', inplace=True)
#fill missing values in the 'address' column with 'Unknown'
data['address'].fillna('Unknown', inplace=True)
#display the first 5 rows of the dataset after filling missing values
print(data.head())
#access the 'name' column of the dataset
name_column = data['name']
print(name_column)
#access the multiple columns 'name' and 'address' of the dataset
name_address_columns = data[['name', 'address']]    
print(name_address_columns)
#filter the dataset to include only rows where the 'name' column is 'McDonald's'    
mcdonalds_data = data[data['name'] == "McDonald's"]
print(mcdonalds_data)
#filter the dataset to include only rows where the 'address' column contains 'New York'
new_york_data = data[data['address'].str.contains('New York')]
print(new_york_data)
#group the dataset by the 'name' column and count the number of occurrences of each unique value
name_counts = data['name'].value_counts()   
print(name_counts)
#selecting the single row using the .loc method
first_row = data.loc[0] 
print(first_row)
#selecting multiple rows using the .loc method  
first_five_rows = data.loc[0:4]
print(first_five_rows)
#selecting a single value using the .loc method 
first_name = data.loc[0, 'name']
print(first_name)
#selecting pieces of data using the .loc method
name_and_address = data.loc[0:4, ['name', 'address']]
print(name_and_address)
#coditional selection using the .loc method
mcdonalds_rows = data.loc[data['name'] == "McDonald's"]
print(mcdonalds_rows)
#selecting single column using the .iloc method
name_column_iloc = data.iloc[:, 0]
print(name_column_iloc)
#selecting multiple columns using the .iloc method
name_address_columns_iloc = data.iloc[:, 0:2]
print(name_address_columns_iloc)
#selecting a single value using the .iloc method
first_name_iloc = data.iloc[0, 0]
print(first_name_iloc)
#selecting pieces of data using the .iloc method
name_and_address_iloc = data.iloc[0:5, 0:2]
print(name_and_address_iloc)
#conditional selection using the .iloc method
mcdonalds_rows_iloc = data.iloc[data['name'] == "McDonald's"]
print(mcdonalds_rows_iloc)
#delete row with index 1
data.drop(index=1, inplace=True)
#display the first 5 rows of the dataset after deleting the row with index 1
print(data.head())
#delete the row with index 2 and 3
data.drop(index=[2, 3], inplace=True)
#display the first 5 rows of the dataset after deleting the row with index 2 and 3
print(data.head())
#display the modified dataframe after deleting the rows with index 1, 2 and 3
print(data)
#delete the 'address' column from the dataset
data.drop(columns='address', inplace=True)
#display the first 5 rows of the dataset after deleting the 'address' column
print(data.head())
#delete the 'name' column from the dataset
data.drop(columns='name', inplace=True) 
#display the  modified dataframe after deleting the 'name' column
print(data)
# rename the 'name' column to 'restaurant_name'
data.rename(columns={'name': 'restaurant_name'}, inplace=True)
#rename the 'address' column to 'restaurant_address'
data.rename(columns={'address': 'restaurant_address'}, inplace=True)
#rename the province and city columns to 'province' and 'city' respectively
data.rename(columns={'province': 'province', 'city': 'city'}, inplace=True)
#display modified dataframe after renaming the columns
print(data.head())
# rename column one index label to 'restaurant_name'
data.rename(columns={data.columns[0]: 'restaurant_name'}, inplace=True)
# rename column two index label to 'restaurant_address'
data.rename(columns={data.columns[1]: 'restaurant_address'}, inplace=True)
#display modified dataframe after renaming the columns using index labels
print(data.head())
#select the row where name is 'Burger King' 
burger_king_row = data.loc[data['restaurant_name'] == 'Burger King']
print(burger_king_row)
#sort the dataset by the 'restaurant_name' column in ascending order
sorted_data_asc = data.sort_values(by='restaurant_name', ascending=True)    
print(sorted_data_asc)
#sort the dataset by the 'restaurant_name' column in descending order
sorted_data_desc = data.sort_values(by='restaurant_name', ascending=False)  
print(sorted_data_desc)
#sort pandas dataframe by multiple columns 'restaurant_name' and 'restaurant_address'
sorted_data_multiple = data.sort_values(by=['restaurant_name', 'restaurant_address'], ascending=[True, True])
print(sorted_data_multiple)
#group the dataset by the 'restaurant_name' column and calculate the mean of each group
grouped_data = data.groupby('restaurant_name')   
print(grouped_data)
#use dropna() method to remove rows with missing values in the dataset
data.dropna(inplace=True)
#use fillna() method to fill missing values in the dataset
data.fillna('Unknown', inplace=True)
#display the first 5 rows of the dataset after handling missing values
print(data.head())