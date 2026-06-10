import numpy as np

brokered_by, price , acre_lot ,house_size = np.genfromtxt('RealEstate-USA.csv', delimiter=',', usecols=(0,2,5,10), unpack=True, dtype=None,skip_header=1)
print("Brokered by: ", brokered_by)
print("Price: ", price)
print("Acre Lot: ", acre_lot)
print("House Size: ", house_size)

# real estate prices - statistics operations
print("Mean price: ", np.mean(price))
print("Median price: ", np.median(price))
print("Standard deviation of price: ", np.std(price))
print("Minimum price: ", np.min(price))
print("Maximum price: ", np.max(price))
print("25th percentile price: ", np.percentile(price, 25))
print("75th percentile price: " , np.percentile(price,75))
print("price average: ", np.average(price))
print("price mode: ", np.bincount(price.astype(int)).argmax())

# real estate prices - mathematical operations
print("Price squared: ", np.square(price))
print("Price cubed: ", np.power(price, 3))
print("price sqrt: ", np.sqrt(price))
print("price pow 2: ", np.power(price, 2))
print("price abs: ", np.abs(price))

# preform basic arithmetic operations on price and acre_lot
print("Price + Acre Lot: ", np.add(price, acre_lot))
print("Price - Acre Lot: ", np.subtract(price, acre_lot))
print("Price * Acre Lot: ", np.multiply(price, acre_lot))
print("Price / Acre Lot: ", np.divide(price, acre_lot))

# trignometric functions on price
# normalize price
price_norm = (price - np.mean(price)) / np.std(price)

# scale to small range
price_scaled = price_norm / np.max(np.abs(price_norm))

# trig functions
sine_values = np.sin(price_scaled)
cosine_values = np.cos(price_scaled)
tangent_values = np.tan(price_scaled)

print(sine_values)
print(cosine_values)
print(tangent_values)

# Calculate the natural logarithm and base-10 logarithm of the price
log_price = np.log(price)
log10_price = np.log10(price)

print("Natural logarithm of price: ", log_price)
print("Base-10 logarithm of price: ", log10_price)

# Calculate the hyperbolic sine of each element in the price array
sinh_price = np.sinh(price)
print("Hyperbolic sine of price: ", sinh_price)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element in the price array
cosh_price = np.cosh(price)
print("Hyperbolic cosine of price: ", cosh_price)

# Hyperbolic Tangent Using tanh() Function
# Calculate the hyperbolic tangent of each element in the price array   
tanh_price = np.tanh(price)
print("Hyperbolic tangent of price: ", tanh_price)

# Calculate the inverse hyperbolic sine of each element in the price array
asinh_price = np.arcsinh(price) 
print("Inverse hyperbolic sine of price: ", asinh_price)

# Calculate the inverse hyperbolic cosine of each element in the price array
acosh_price = np.arccosh(price) 
print("Inverse hyperbolic cosine of price: ", acosh_price)

# real estate house size plus acre lot - 2 dementional array
house_size_plus_acre_lot = np.add(house_size, acre_lot)
print("House size plus acre lot: ", house_size_plus_acre_lot)

# check the dimensions of the array1
print("Dimensions of house_size_plus_acre_lot: ", house_size_plus_acre_lot.shape)

# return the total number of elements in the array
print("Total number of elements in house_size_plus_acre_lot: ", house_size_plus_acre_lot.size)

# return the a tuple that gives size of array in each dimension
print("Shape of house_size_plus_acre_lot: ", house_size_plus_acre_lot.shape)

# check the data type of array1
print("Data type of house_size_plus_acre_lot: ", house_size_plus_acre_lot.dtype)    

# slicing array
print("First 5 elements of price: ", price[:5])
print("Last 5 elements of price: ", price[-5:])
print("Elements 5 to 10 of price: ", price[5:10])

# indexing array
print("Price at index 0: ", price[0])
print("Price at index 10: ", price[10])
print("Price at index -1: ", price[-1])

#Iterating over price array using nditer
print("Iterating over price array using nditer:")
for element in np.nditer(price):
    print(element)

#EDIT: If you need indexes (as a tuple for 2D table), then:
print("Iterating over price array with indexes using ndenumerate:")
for index, element in np.ndenumerate(price):
    print(index, element)

# reshaping price array to 1, -1 ( automatically calculate the number of columns based on the original size of the array)
reshaped_price = price.reshape(1, -1)
print("Reshaped price (1 x 298): ", reshaped_price)
print()