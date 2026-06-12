import pandas as pd
# Load the dataset
data = pd.read_csv('startup_growth_investment_data.csv')
# Display the first few rows of the dataset to verify the changes
print(data.head())
# display the last few rows of the dataset to verify the changes
print(data.tail())
# display the types of the columns in the dataset to verify the changes
print(data.dtypes)
# display the summary statistics of the dataset to verify the changes
print(data.describe())
# missing values in the dataset to verify the changes
print(data.isnull().sum())
# check for duplicates in the dataset to verify the changes
print(data.duplicated().sum())
# select single column using .loc 
print(data.loc[:, 'Industry'])
# select multiple columns using .loc
print(data.loc[:, ['Funding Rounds']])
# select slice of rows and columns using .loc
print(data.loc[0:5, 'Industry':'Funding Rounds'])
# conditional selection using .loc
print(data.loc[data['Funding Rounds'] > 3, ['Startup Name', 'Funding Rounds']])
# select single row using .loc
print(data.loc[0])
# select multiple rows using .loc
print(data.loc[0:5])
# select slice of rows using .loc
print(data.loc[0:5])
# select single column using .iloc
print(data.iloc[:, 2])
# select multiple columns using .iloc
print(data.iloc[:, [2, 3]])
# select slice of rows and columns using .iloc
print(data.iloc[0:5, 2:4])
# conditional selection using .iloc
print(data.iloc[data['Funding Rounds'] > 3, [0, 2]])
# select single row using .iloc
print(data.iloc[0])
# select multiple rows using .iloc
print(data.iloc[0:5])
# select slice of rows using .iloc
print(data.iloc[0:5])
# query the dataset to find startups in the 'Technology' country
technology_startups = data[data['Country'] == 'Technology']
# query the dataset to find startups founded after 2010
recent_startups = data[data['Year Founded'] > 2010]
# query the dataset to find startups with more than 3 funding rounds
well_funded_startups = data[data['Funding Rounds'] > 3]
# display the results of the queries
print("Technology Startups:")
print(technology_startups)
print("Recent Startups:")
print(recent_startups)
print("Well-Funded Startups:")
print(well_funded_startups)
#sort the dataset by 'Year Founded' in ascending order
sorted_data = data.sort_values(by='Year Founded')
print("Sorted by Year Founded (Ascending):")
print(sorted_data)
#sort the dataset by 'Funding Rounds' in descending order
sorted_data_desc = data.sort_values(by='Funding Rounds', ascending=False)
print("Sorted by Funding Rounds (Descending):")
print(sorted_data_desc)
# group the dataset by 'Country' and calculate the average number of funding rounds for each country
average_funding_by_country = data.groupby('Country')['Funding Rounds'].mean()
print("Average Funding Rounds by Country:")
print(average_funding_by_country)
# group the dataset by 'Year Founded' and calculate the total valuation for each year
total_valuation_by_year = data.groupby('Year Founded')['Valuation (USD)'].sum()
print("Total Valuation by Year Founded:")
print(total_valuation_by_year)
# group the dataset by 'Industry' and calculate the maximum valuation for each industry
max_valuation_by_industry = data.groupby('Industry')['Valuation (USD)'].max()
print("Maximum Valuation by Industry:")
print(max_valuation_by_industry)
# drop the 'Valuation (USD)' column from the dataset
data_without_valuation = data.drop(columns=['Valuation (USD)'])
print("Dataset without Valuation (USD) column:")
print(data_without_valuation)
# drop rows with missing values in the 'Funding Rounds' column
data_without_missing_funding = data.dropna(subset=['Funding Rounds'])   
print("Dataset without missing values in Funding Rounds column:")
print(data_without_missing_funding)
# fill missing values in the 'Year Founded' column with the mean year
mean_year_founded = data['Year Founded'].mean()
data_filled_year_founded = data.copy()
data_filled_year_founded['Year Founded'] = data_filled_year_founded['Year Founded'].fillna(mean_year_founded)
print("Dataset with missing values in Year Founded column filled with mean:")
print(data_filled_year_founded)
# delete row with index 0
data_dropped_row = data.drop(index=0)
print("Dataset with row index 0 dropped:")
print(data_dropped_row)
# delete column 'Industry'
data_dropped_column = data.drop(columns=['Industry'])
print("Dataset with 'Industry' column dropped:")
print(data_dropped_column)
# rename column 'Valuation (USD)' to 'Valuation'
data_renamed_column = data.rename(columns={'Valuation (USD)': 'Valuation'}) 
print("Dataset with 'Valuation (USD)' column renamed to 'Valuation':")
print(data_renamed_column)
# rename column 'Year Founded' to 'Founded Year'
data_renamed_column_year = data.rename(columns={'Year Founded': 'Founded Year'})
print("Dataset with 'Year Founded' column renamed to 'Founded Year':")
print(data_renamed_column_year)
# rename multiple columns
data_renamed_columns = data.rename(columns={'Valuation (USD)': 'Valuation', 'Year Founded': 'Founded Year'})
print("Dataset with multiple columns renamed:") 
print(data_renamed_columns)
# drop duplicates from the dataset
data_no_duplicates = data.drop_duplicates()
print("Dataset with duplicates dropped:")
print(data_no_duplicates)
# data cleaning: remove leading and trailing whitespace from the 'Startup Name' column
data_cleaned = data.copy()
data_cleaned['Startup Name'] = data_cleaned['Startup Name'].str.strip()
print("Dataset with leading and trailing whitespace removed from 'Startup Name' column:")
print(data_cleaned)