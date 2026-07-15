import pandas as pd

# Load the dataset
data = pd.read_csv('real_estate_sales_2001-2022_gl-short.csv')

# Display the first few rows of the dataset
print(data.head())
#display the last few rows of the dataset
print(data.tail())
# diplay the number of rows and columns in the dataset
print(data.shape)
# display the column names of the dataset
print(data.columns)
# display the summary information of the dataset
print(data.info())
# display the summary statistics of the dataset 
print(data.describe())
# display the data types of each column
print(data.dtypes)
# Check for missing values
print(data.isnull().sum())
# Check for duplicate rows
print(data.duplicated().sum())
# Display the unique values in the 'Property Type' column
print(data['Property Type'].unique())
# Display the unique values in the 'Date Recorded' column
print(data['Date Recorded'].unique()) 

# select single column using .loc
property_type = data.loc[:, 'Property Type']
print(property_type.head())

# select multiple columns using .loc
selected_columns = data.loc[:, ['Property Type', 'Sale Amount']]
print(selected_columns.head())

# select columns based on a condition using .loc
high_value_sales = data.loc[data['Sale Amount'] > 1000000, ['Property Type', 'Sale Amount']]
print(high_value_sales.head())

# select slice of rows and columns using .loc
slice_data = data.loc[10:20, ['Property Type', 'Sale Amount']]
print(slice_data)

# select single row using .loc
single_row = data.loc[5]
print(single_row)

# select multiple rows using .loc
multiple_rows = data.loc[5:10]
print(multiple_rows)

# select rows based on a condition using .loc
high_value_sales_rows = data.loc[data['Sale Amount'] > 1000000]
print(high_value_sales_rows.head())

# select slice of rows using .loc
slice_rows = data.loc[10:20]
print(slice_rows)

# select single column using .iloc
property_type_iloc = data.iloc[:, 2]  # Assuming 'Property Type' is the third column
print(property_type_iloc.head())

# select multiple columns using .iloc
selected_columns_iloc = data.iloc[:, [2, 5]]  # Assuming 'Property Type' is the third column and 'Sale Amount' is the sixth column
print(selected_columns_iloc.head())

# select columns based on a condition using .iloc
high_value_sales_iloc = data.iloc[data['Sale Amount'] > 1000000, [2, 5]]  # Assuming 'Property Type' is the third column and 'Sale Amount' is the sixth column
print(high_value_sales_iloc.head())

# select slice of rows and columns using .iloc
slice_data_iloc = data.iloc[10:20, [2, 5]]  # Assuming 'Property Type' is the third column and 'Sale Amount' is the sixth column
print(slice_data_iloc)      

# select single row using .iloc
single_row_iloc = data.iloc[5]
print(single_row_iloc)

# select multiple rows using .iloc
multiple_rows_iloc = data.iloc[5:10]
print(multiple_rows_iloc)

# select rows based on a condition using .iloc
high_value_sales_rows_iloc = data.iloc[data['Sale Amount'] > 1000000]
print(high_value_sales_rows_iloc.head())

# select slice of rows using .iloc
slice_rows_iloc = data.iloc[10:20]
print(slice_rows_iloc)

#qurey the dataset to find all sales of 'Single Family' properties
single_family_sales = data.loc[data['Property Type'] == 'Single Family']    
print(single_family_sales.head())

#query the dataset to find all sales that occurred in the year 2020
sales_2020 = data.loc[data['Date Recorded'].str.contains('2020')]
print(sales_2020.head())

#filter the dataset to find all sales with a sale amount greater than $500,000
high_value_sales_filter = data.loc[data['Sale Amount'] > 500000]    
print(high_value_sales_filter.head())

#sort the dataset by 'Sale Amount' in descending order
sorted_sales = data.sort_values(by='Sale Amount', ascending=False)
print(sorted_sales.head())

#sort the dataset by 'Date Recorded' in ascending order
sorted_by_date = data.sort_values(by='Date Recorded', ascending=True)   
print(sorted_by_date.head())

#group the dataset by 'Property Type' and calculate the average sale amount for each property type
average_sale_by_property_type = data.groupby('Property Type')['Sale Amount'].mean() 
print(average_sale_by_property_type)

#group the dataset by 'Date Recorded' and calculate the total sale amount for each date
total_sale_by_date = data.groupby('Date Recorded')['Sale Amount'].sum()
print(total_sale_by_date)

#delete row with index 1
data_dropped_row = data.drop(index=1)
print(data_dropped_row.head())

#delete column 'Sale Amount'
data_dropped_column = data.drop(columns='Sale Amount')
print(data_dropped_column.head())

#delete rows with missing values
data_dropped_missing = data.dropna()
print(data_dropped_missing.head())

#delete duplicate rows  
data_dropped_duplicates = data.drop_duplicates()
print(data_dropped_duplicates.head())

#rename column 'Sale Amount' to 'Price'
data_renamed_column = data.rename(columns={'Sale Amount': 'Price'})
print(data_renamed_column.head())

#rename column 'Date Recorded' to 'Sale Date'
data_renamed_date = data.rename(columns={'Date Recorded': 'Sale Date'}) 
print(data_renamed_date.head())

#fill missing values in 'Sale Amount' column with the mean sale amount
mean_sale_amount = data['Sale Amount'].mean()   
data_filled_missing = data.copy()
data_filled_missing['Sale Amount'] = data_filled_missing['Sale Amount'].fillna(mean_sale_amount)
print(data_filled_missing.head())

#data cleaning: remove rows with missing values in 'Sale Amount' column
data_cleaned = data.dropna(subset=['Sale Amount'])
print(data_cleaned.head())
