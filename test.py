import numpy as np


# create a 3x4 matrix with numbers 1 to 12

mat_a = np.arange(1, 9).reshape(4, 2)
mat_b = np.arange(1, 19).reshape(2, 9)

mat_c = mat_a @ mat_b

print(mat_c)
