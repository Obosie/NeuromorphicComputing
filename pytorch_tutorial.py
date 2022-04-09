# Within your Virtual Environemt, Pytorch should have been downloaded
# Make sure the interpreter path points to that of your Virtual Environment
#   If not; this code will fail to run. 

from turtle import clear
import torch
import numpy as np

'''
# Tensors can be created from data, numPy arrays and from other tensors
# Created from data 
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)

# Created from numPy arrays
np_array = np.array(data)
print("This is nparray: \n", np_array)
x_np = torch.from_numpy(np_array)
print("This is tensor from nparray: \n",x_np)


# EMPTY TENSORS 
# Dimensions: row x column
# prints an "empty" 1D tensor with 1 element; value not intialized -- will be random
x = torch.empty(1)
print(x)
# prints an "empty" 1D tensor with 3 elements; value not intialized -- will be random
x = torch.empty(3)
print(x)
# prints a 2x3 matrix tensor, 2 D matrix
x = torch.empty(2,3)
print(x)

# RANDOMIZED TENSORS
x = torch.rand(1,3)
print(x)

# tensors filled with specifcally ones or zeroes 
x = torch.ones(1,3)
print(x)
x = torch.zeros(1,3)
print(x)

# getting and changing tensor data types
x = torch.ones(1,3)
print(x.dtype)

# give specicy dtype -- changed torch values to int
x = torch.ones(1,3, dtype=torch.int)
print(x)
print(x.dtype)

# Functionality between two tensors 
x = torch.rand(2,2)
print(x)
y = torch.rand(2,2)
print(y)
z = x + y
# adds each element in each tensor then creates z
print(z)


# Splicing 
print("Pre slice: \n")
x = torch.rand(5,3)
print(x)
print("\n")
print("Post slice: \n")
# everything up to col 0
print(x[:,0])
# everything up to row 0
print(x[0,:])
# specific element
print(x[0,0])
'''

# Resizing -- use .view

# Converting from Tensor to Numpy 
a = torch.ones(5, dtype=torch.int)
print(a)
b = a.numpy()
print(type(b))

# Converting from Numpy to Tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)