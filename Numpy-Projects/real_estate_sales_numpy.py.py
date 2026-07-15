import numpy as np

Serial_Number, Assessed_Value, Sale_Amount , Sales_Ratio= np.genfromtxt('Real_Estate_Sales_2001-2022_GL-Short.csv', delimiter=',', usecols=(0,5,6,7), unpack=True, dtype=None,skip_header=1, invalid_raise=False)
print("Serial_Number:", Serial_Number)
print("Assessed_Value:", Assessed_Value)
print("Sale_Amount:", Sale_Amount)  
print("Sales_Ratio:", Sales_Ratio)

# real estate sales sale_amount - statistics operations
print("Mean of Sale_Amount:", np.mean(Sale_Amount))
print("Median of Sale_Amount:", np.median(Sale_Amount))
print("Standard Deviation of Sale_Amount:", np.std(Sale_Amount))
print("Variance of Sale_Amount:", np.var(Sale_Amount))
print("Minimum of Sale_Amount:", np.min(Sale_Amount))
print("Maximum of Sale_Amount:", np.max(Sale_Amount))   
print("25th Percentile of Sale_Amount:", np.percentile(Sale_Amount, 25))
print("50th Percentile of Sale_Amount:", np.percentile(Sale_Amount, 50))
print("75th Percentile of Sale_Amount:", np.percentile(Sale_Amount, 75))    

# real estate sales assessed_value - maths operations
print("real estate sales assessed_value:square", np.square(Assessed_Value))
print("real estate sales assessed_value:square root", np.sqrt(Assessed_Value))
print("real estate sales assessed_value:power", np.power(Assessed_Value, 2))
print("real estate sales assessed_value:abs", np.abs(Assessed_Value))

#preform basic arithmetic operations on real estate sales sale_amount and assessed_value
print("real estate sales sale_amount + assessed_value:", np.add(Sale_Amount, Assessed_Value))
print("real estate sales sale_amount - assessed_value:", np.subtract(Sale_Amount, Assessed_Value))
print("real estate sales sale_amount * assessed_value:", np.multiply(Sale_Amount, Assessed_Value))
print("real estate sales sale_amount / assessed_value:", np.divide(Sale_Amount, Assessed_Value))

# trigonometric functions on real estate sales sales_ratio
print("real estate sales sales_ratio:sine", np.sin(Sales_Ratio))
print("real estate sales sales_ratio:cosine", np.cos(Sales_Ratio)) 
print("real estate sales sales_ratio:tangent", np.tan(Sales_Ratio))

# calculate the natural logarithm of real estate sales sale_amount
print("real estate sales sale_amount:natural logarithm", np.log(Sale_Amount))
print("real estate sales sale_amount:log base 10", np.log10(Sale_Amount))

#calculate the hyperbolic sine of each element
sinh_values = np.sinh(Sale_Amount)
print("real estate sales sale_amount:hyperbolic sine", sinh_values)

# calculate the hyperbolic cosine of each element
cosh_values = np.cosh(Sale_Amount)  
print("real estate sales sale_amount:hyperbolic cosine", cosh_values)

# calculate the hyperbolic tangent of each element
tanh_values = np.tanh(Sale_Amount)
print("real estate sales sale_amount:hyperbolic tangent", tanh_values)

# calculate the inverse hyperbolic cosine of each element
arcosh_values = np.arccosh(Sale_Amount)
print("real estate sales sale_amount:inverse hyperbolic cosine", arcosh_values) 

# real estate sales sale_amount plus assessed_value - 2 dimensional array
d2sale_amountAssessed_Value = np.array([Sale_Amount, Assessed_Value])
print("real estate sales sale_amount plus assessed_value - 2 dimensional array:", d2sale_amountAssessed_Value)

# check the dimension of array1
print("Dimension of d2sale_amountAssessed_Value:", d2sale_amountAssessed_Value.shape)

# return total number of elements in the array
print("Total number of elements in d2sale_amountAssessed_Value:", d2sale_amountAssessed_Value.size) 

#return a tuple that gives the size of the array along each dimension
print("Shape of d2sale_amountAssessed_Value:", d2sale_amountAssessed_Value.shape)

# check the data type of the array
print("Data type of d2sale_amountAssessed_Value:", d2sale_amountAssessed_Value.dtype)

# slicing array
d2sale_amountAssessed_Value_slice = d2sale_amountAssessed_Value[:, :5]  # slice the first 5 columns
print("Sliced d2sale_amountAssessed_Value (first 5 columns):", d2sale_amountAssessed_Value_slice)   
d2sale_amountAssessed_Value_slice2 = d2sale_amountAssessed_Value[0, :5]  # slice the first 5 elements of the first row
print("Sliced d2sale_amountAssessed_Value (first 5 elements of the first row):", d2sale_amountAssessed_Value_slice2)    

# indexing array
print("Element at index [0, 0] (first row, first column):", d2sale_amountAssessed_Value[0, 0])  # first row, first column
print("Element at index [1, 0] (second row, first column):", d2sale_amountAssessed_Value[1, 0])  # second row, first column
print("Element at index [0, 1] (first row, second column):", d2sale_amountAssessed_Value[0, 1])  # first row, second column
print("Element at index [1, 1] (second row, second column):", d2sale_amountAssessed_Value[1, 1])  # second row, second column   

# you should use the builtin function nditer, if you dont need to have the indexes value
print("Iterating over d2sale_amountAssessed_Value using nditer:")
for element in np.nditer(d2sale_amountAssessed_Value):
    print(element)  
# you should use the builtin function ndenumerate, if you need to have the indexes value
print("Iterating over d2sale_amountAssessed_Value using ndenumerate:")  
for index, element in np.ndenumerate(d2sale_amountAssessed_Value):
    print("Index:", index, "Element:", element)

# reshaping d2_array to 1 row and -1 column (automatically calculated)
d2sale_amountAssessed_Value_reshaped = d2sale_amountAssessed_Value.reshape(1, -1)
print("Reshaped d2sale_amountAssessed_Value (1 row, -1 column):", d2sale_amountAssessed_Value_reshaped)
print("Shape of reshaped d2sale_amountAssessed_Value:", d2sale_amountAssessed_Value_reshaped.shape) 