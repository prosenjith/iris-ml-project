import numpy as np


def create_array():
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)

    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr_2d)

    zeros = np.zeros((3, 3))
    print(zeros)

    ones = np.ones((2, 4))
    print(ones)

    arr_range = np.arange(0, 10, 2)  # Start, Stop, Step
    print(arr_range)

    linear_space = np.linspace(0, 1, 5)
    print(linear_space)

    print(arr.ndim)  # Number of dimensions
    print(arr.shape)  # Shape of the array
    print(arr.size)  # Total number of elements
    print(arr.dtype)  # Data type of elements
    print(arr_2d[:, 1])  # All rows, second column

    random_arr = np.random.rand(3)  # 1D array with 3 random floats
    random_matrix = np.random.rand(2, 3)  # 2x3 random matrix
    random_ints = np.random.randint(0, 10, size=(3, 3))  # 3x3 matrix of random integers
    print("Randon array:\n", random_arr)
    print("Randon matrix:\n", random_matrix)
    print("Randon Ints:\n", random_ints)


if __name__ == "__main__":
    create_array()
