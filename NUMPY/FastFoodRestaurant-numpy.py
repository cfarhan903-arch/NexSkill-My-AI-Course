import numpy as np 

longitude, lat, name = np.genfromtxt('FastFoodRestaurants.csv', delimiter=',', usecols=(4,5,6), unpack=True, dtype=('f8','f8','U100'), encoding='utf-8', skip_header=1, invalid_raise=False)

# fastfood-lat-statistics opperations
print("fast food restaurant lat mean:",np.mean(lat))
print("fast food restaurant lat average:",np.average(lat))
print("fast food restaurant lat std:",np.std(lat))
print("fast food restaurant lat mod:",np.median(lat))
print("fast food restaurant lat percentile - 25:",np.percentile(lat,25))
print("fast food restaurant lat percentile - 75:",np.percentile(lat,75))
print("fast food restaurant lat percentile -  3:",np.percentile(lat,3))
print("fast food restaurant lat min:",np.min(lat))
print("fast food restaurant lat max:",np.max(lat))

# fastfood-longitude-maths opperations  
print("fast food restaurant longitude square:",np.square(longitude))
print("fast food restaurant longitude sqrt:",np.sqrt(longitude))
print("fast food restaurant longitude abs:",np.abs(longitude))

# preform basic airthmetic opperation
print("fast food restaurant longitude + lat:",np.add(longitude,lat))
print("fast food restaurant longiyude - lat:",np.subtract(longitude,lat))
print("fast food restaurant longitude * lat:",np.multiply(longitude,lat))
print("fast food restaurant longitude / lat:",np.divide(longitude,lat))

# trigonometric function
print("fast food restaurant longitude:sine", np.sin(longitude))
print("fast food restaurant longitude:cosine", np.cos(longitude)) 
print("fast food restaurant longitude:tangent", np.tan(longitude))

# calculate the natural logarithm 
print("fast food restaurant longitude:natural logarithm", np.log(longitude))
print("fast food restaurant longitude:log base 10", np.log10(longitude))

#calculate the hyperbolic sine of each element
sinh_values = np.sinh(longitude)
print("fast food restaurant longitude:hyperbolic sine", sinh_values)

# calculate the hyperbolic cosine of each element
cosh_values = np.cosh(longitude)  
print("fast food restaurant longitude:hyperbolic cosine", cosh_values)

# calculate the hyperbolic tangent of each element
tanh_values = np.tanh(longitude)
print("fast food restayrant longitude:hyperbolic tangent", tanh_values)

# calculate the inverse hyperbolic cosine of each element
arcosh_values = np.arccosh(longitude)
print("fast food restaurant longitude:inverse hyperbolic cosine", arcosh_values) 

# longitude and lat - 2 dimensional array
d2longitude_lat = np.array([longitude, lat])
print("fast food restaurant longitude plus lat - 2 dimensional array:", d2longitude_lat)

# check the dimension of array1
print("Dimension of d2longitude_lat:", d2longitude_lat.shape)

# return total number of elements in the array
print("Total number of elements in d2longitude_lat:", d2longitude_lat.size) 

#return a tuple that gives the size of the array along each dimension
print("Shape of d2longitude_lat:", d2longitude_lat.shape)

# check the data type of the array
print("Data type of d2longitude_lat:", d2longitude_lat.dtype)

# slicing array
d2longitude_lat_slice = d2longitude_lat[:, :5]  # slice the first 5 columns
print("Sliced d2longitude_lat (first 5 columns):", d2longitude_lat_slice)   
d2longitude_lat_slice2 = d2longitude_lat[0, :5]  # slice the first 5 elements of the first row
print("Sliced d2longitude_lat (first 5 elements of the first row):", d2longitude_lat_slice2)    

# indexing array
print("Element at index [0, 0] (first row, first column):", d2longitude_lat[0, 0])  # first row, first column
print("Element at index [1, 0] (second row, first column):", d2longitude_lat[1, 0])  # second row, first column
print("Element at index [0, 1] (first row, second column):", d2longitude_lat[0, 1])  # first row, second column
print("Element at index [1, 1] (second row, second column):", d2longitude_lat[1, 1])  # second row, second column   

# you should use the builtin function nditer, if you dont need to have the indexes value
print("Iterating over d2longitude_lat using nditer:")
for element in np.nditer(d2longitude_lat):
    print(element)  
# you should use the builtin function ndenumerate, if you need to have the indexes value
print("Iterating over d2longitude_lat using ndenumerate:")  
for index, element in np.ndenumerate(d2longitude_lat):
    print("Index:", index, "Element:", element)

# reshaping d2_array to 1 row and -1 column (automatically calculated)
d2longitude_lat_reshaped = d2longitude_lat.reshape(1, -1)
print("Reshaped d2longitude_lat (1 row, -1 column):", d2longitude_lat_reshaped)
print("Shape of reshaped d2longitude_lat:", d2longitude_lat_reshaped.shape) 