import numpy as np

funding_rounds, investment_amount_usd, valuation_usd, number_of_investors, growth_rate = np.genfromtxt(
	'startup_growth_investment_data.csv',
	delimiter=',',
	usecols=(2, 3, 4, 5, 8),
	unpack=True,
	dtype=None,
	skip_header=1,
)

print("Funding Rounds:", funding_rounds)
print("Investment Amount (USD):", investment_amount_usd)
print("Valuation (USD):", valuation_usd)
print("Number of Investors:", number_of_investors)
print("Growth Rate (%):", growth_rate)

# startup_growth_investment_data investment_amount_usd- statistics operations
print("Mean Investment Amount (USD):", np.mean(investment_amount_usd))
print("Median Investment Amount (USD):", np.median(investment_amount_usd))
print("Average Investment Amount (USD):", np.average(investment_amount_usd))
print("Standard Deviation of Investment Amount (USD):", np.std(investment_amount_usd))
print("Variance of Investment Amount (USD):", np.var(investment_amount_usd))
print("Minimum Investment Amount (USD):", np.min(investment_amount_usd))
print("Maximum Investment Amount (USD):", np.max(investment_amount_usd))
print("25th Percentile of Investment Amount (USD):", np.percentile(investment_amount_usd, 25))
print("50th Percentile of Investment Amount (USD):", np.percentile(investment_amount_usd, 50))
print("75th Percentile of Investment Amount (USD):", np.percentile(investment_amount_usd, 75))

# startup_growth_investment_data number_of_investors - mathematical operations
print("number_of_investors squared:", np.square(number_of_investors))
print("number_of_investors sqrt:", np.sqrt(number_of_investors))
print("number_of_investors power of 3:", np.power(number_of_investors, 3))
print("number_of_investors abs: " , np.abs(number_of_investors))

# perform basic arithmetic operations between funding_rounds and growth_rate
print("funding_rounds + growth_rate:", funding_rounds + growth_rate)
print("funding_rounds - growth_rate:", funding_rounds - growth_rate)
print("funding_rounds * growth_rate:", funding_rounds * growth_rate)
print("funding_rounds / growth_rate:", funding_rounds / growth_rate)

#Trigonometric Functions
print("Sine of growth_rate:", np.sin(growth_rate))
print("Cosine of growth_rate:", np.cos(growth_rate))    
print("Tangent of growth_rate:", np.tan(growth_rate))

# Calculate the natural logarithm and base-10 logarithm of valuation_usd
print("Natural logarithm of valuation_usd:", np.log(valuation_usd)) 
print("Base-10 logarithm of valuation_usd:", np.log10(valuation_usd))

# Calculate the hyperbolic sine of each element in growth_rate
sinh_growth_rate = np.sinh(growth_rate)
print("Hyperbolic sine of growth_rate:", sinh_growth_rate)

# Calculate the hyperbolic cosine of each element in growth_rate
cosh_growth_rate = np.cosh(growth_rate)
print("Hyperbolic cosine of growth_rate:", cosh_growth_rate)

# Calculate the hyperbolic tangent of each element in growth_rate
tanh_growth_rate = np.tanh(growth_rate) 
print("Hyperbolic tangent of growth_rate:", tanh_growth_rate)

# Calculate the inverse hyperbolic sine of each element in growth_rate
asinh_growth_rate = np.arcsinh(growth_rate) 
print("Inverse hyperbolic sine of growth_rate:", asinh_growth_rate)

# Calculate the inverse hyperbolic cosine of each element in growth_rate
acosh_growth_rate = np.arccosh(growth_rate)
print("Inverse hyperbolic cosine of growth_rate:", acosh_growth_rate)   

#Startup_growth_investment_data growth_rate plus funding_rounds  - 2 dimensional array
d2_array = np.array([growth_rate, funding_rounds])
print("2D Array of growth_rate and funding_rounds:\n", d2_array)

# check the dimension of array1
print("Dimension of d2_array:", d2_array.ndim)

# return total number of elements in array1
print("Total number of elements in d2_array:", d2_array.size)

# return a tuple that gives size of array in each dimension
print("Shape of d2_array:", d2_array.shape)

# check the data type of array1
print("Data type of d2_array:", d2_array.dtype)

# Splicing array
d2_array_slice = d2_array[0:1, 0:3]  # Slicing the first row and first three columns
print("Sliced 2D Array:\n", d2_array_slice)
d2_array_slice2 = d2_array[1:2, 0:3]  # Slicing the second row and first three columns
print("Sliced 2D Array:\n", d2_array_slice2)

# indexing array
print("Element at index [0, 0] (growth_rate of first startup):", d2_array[0, 0])
print("Element at index [1, 0] (funding_rounds of first startup ):", d2_array[1, 0])    

# Iterate through each element in d2_array using nditer
print("Iterating through d2_array using nditer:")
for element in np.nditer(d2_array):
    print(element)
# You can also use ndenumerate to get both the index and the value of each element in the array.
print("Iterating through d2_array using ndenumerate:")
for index, element in np.ndenumerate(d2_array):
    print("Index:", index, "Value:", element)

# Reshaping d2_array to 1 row and -1 columns (automatically calculated)
reshaped_array = d2_array.reshape(1, -1)
print("Reshaped Array (1 row, -1 columns):\n", reshaped_array)
print()