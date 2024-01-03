
import os
from matplotlib.pylab import PCG64

import numpy as np
import kfoldLog
import readMonkAndCup as readMC

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# dirName = BASE_DIR+"/CupDatasets/Cup23"
result = np.array(['ciao', 'come', 'stai', 'a', 'v', 'bv'])
# kfoldLog.ML_Cup_Template(result, dirName, 'puppa', '0000000')


# print(result.shape)
# listId = []
# for i in range(1, len(result)+1):
#     print(i)
#     listId.append(i)
# # print(np.array(listId))
# print(np.array(listId).T.shape)
# print(np.concatenate((np.array(listId).reshape((6,1)), np.array(result).reshape((6,1))), axis=1))
# result2 = result.copy()
# result
# np.random.shuffle(result2)
# print(result)
# print(result2)
# arr1inds = result.argsort()
# print(arr1inds)
# print(result[arr1inds[::-1]])
# print(result2[arr1inds[::-1]])
# sorted_arr1 = arr1[arr1inds[::-1]]
# sorted_arr2 = arr2[arr1inds[::-1]]


# import numpy as np

# # Example arrays
# array1 = np.array([1, 3, 2, 4, 5])
# array2 = np.array([1, 3, 2, 4, 5])

# # Get the indices that would sort array1
# indices = np.argsort(array1)
# # Shuffle the second array using the indices

# # Reorder the shuffled array2 based on the original order of array1

# print("Original array1:", array1)
# print("Original array2:", array2)
# permutation_indices = np.random.shuffle(array2)
# print(permutation_indices)
# shuffled_array2 = array2[permutation_indices]
# reordered_array2 = shuffled_array2[np.argsort(permutation_indices)]

# print("Shuffled array2:", shuffled_array2)
# print("Reordered array2:", reordered_array2)


# Original arrays
# array1 = np.array([1, 3, 4, 2, 5])
# array2 = np.array([2, 4, 3, 5, 1])

# # Get the indices that would sort array1
# original_order_indices = np.argsort(array1)
# print(original_order_indices)

# # Reorder array2 based on the original order of array1
# reordered_array2 = array2[original_order_indices]

# print("Original array1:", array1)
# print("Original array2:", array2)
# print("Reordered array2:", reordered_array2)

# print(np.random.Generator(PCG64()).random((3,2) < 0.5)).astype(float)
print((np.random.Generator(PCG64()).random(3) < 0.2).astype(float))
# print((np.random.rand(3) < (0.5)).astype(float))