import numpy as np

# --- Neural Network functions --- #
def get_initial_params(layer_sizes: list[int]):
  if isinstance(layer_sizes, list) and isinstance(layer_sizes[0], int):
    return np.random.rand(*layer_sizes)
  else:
    raise ValueError("`layer_sizes` must be of type `list[int]`")

def one_hot_encode(X):
  one_hot_vector = np.zeros((X.size, X.max() + 1))
  one_hot_vector[np.arange(X.size), X] = 1
  return one_hot_vector.T

def get_accuracy(Y, Y_actual):
  return np.sum(Y == Y_actual) / Y.size