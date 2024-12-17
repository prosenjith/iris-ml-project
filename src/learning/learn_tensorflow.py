import tensorflow as tf


def _creating_tensors():
    scalar = tf.constant(5)
    print("Scalar:", scalar)

    vector = tf.constant([1, 2, 3])
    print("Vector:", vector)

    matrix = tf.constant([[1, 2], [3, 4]])
    print("Matrix:", matrix)

    # Higher-dimensional tensor
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("3D Tensor:", tensor_3d)


def _check_tensor_properties():
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("Shape:", tensor_3d.shape)
    print("Data Type:", tensor_3d.dtype)
    print("Rank (Number of dimensions):", tf.rank(tensor_3d))


def _check_basic_tensor_operations():
    # Create tensors
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])

    # Basic operations
    add = tf.add(a, b)
    sub = tf.subtract(a, b)
    mul = tf.multiply(a, b)

    print("Addition:\n", add)
    print("Subtraction:\n", sub)
    print("Multiplication:\n", mul)

    # Matrix multiplication
    matmul = tf.matmul(a, b)
    print("Matrix Multiplication:\n", matmul)

    reshaped = tf.reshape(a, (4, 1))
    print("Reshaped Tensor:\n", reshaped)


def _check_special_tensor_operations():
    zeros = tf.zeros([3, 3])
    ones = tf.ones([2, 3])

    print("Zeros Tensor:\n", zeros)
    print("Ones Tensor:\n", ones)


def _generate_random_values_tensor():
    random_tensor = tf.random.normal(shape=(3, 3), mean=0, stddev=1)
    print("Random Tensor:\n", random_tensor)

def _slice_and_index_tensor():
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original Tensor:\n", tensor)

    # Slicing the first two rows and first two columns
    slice = tensor[:2, :2]
    print("Sliced Tensor:\n", slice)

if __name__ == "__main__":
    print("Tensorflow Version:", tf.__version__)
    # _creating_tensors()
    # _check_tensor_properties()
    # _check_basic_tensor_operations()
    #_check_special_tensor_operations()
    _slice_and_index_tensor()