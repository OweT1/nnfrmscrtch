import numpy as np

# --- Activation Functions --- #
def Linear(x: np.ndarray):
  return x

def ReLU(x: np.ndarray):
  return np.maximum(0, x)

def LeakyReLU(x: np.ndarray, alpha=0.05):
  def operation(x: int):
    return x if x > 0 else alpha * x
  return np.vectorize(operation)(x)
  
def Sigmoid(x: np.ndarray):
  return 1 / (1 + np.exp(-x))

def Softmax(x: np.ndarray):
  return np.exp(x) / np.sum(np.exp(x))

activation_function_mapping = {
  "linear": Linear,
  "relu": ReLU,
  "leakyrelu": LeakyReLU,
  "sigmoid": Sigmoid,
  "softmax": Softmax
}

# --- Activation Function Derivatives --- #
def LinearDerivative(x: np.ndarray):
  return 1

def ReLUDerivative(x: np.ndarray):
  return x > 0

def LeakyReLUDerivative(x: np.ndarray, alpha=0.05):
  def operation(x: int):
    return 1 if x > 0 else alpha
  return np.vectorize(operation)(x)
  
def SigmoidDerivative(x: np.ndarray):
  return Sigmoid(x) * (1 - Sigmoid(x))

activation_function_derivatives_mapping = {
  "linear": LinearDerivative,
  "relu": ReLUDerivative,
  "leakyrelu": LeakyReLUDerivative,
  "sigmoid": SigmoidDerivative,
}

# --- Activation Function helper --- #
def get_valid_activation_functions():
  return activation_function_mapping.keys()

def validate_activation_function(act_fn: str):
  return act_fn.lower() in get_valid_activation_functions()

def get_activation_function(act_fn: str):
  if validate_activation_function(act_fn):
    return activation_function_mapping[act_fn.lower()]
  else:
    raise ValueError('{} is not a supported activation function.', act_fn)

def get_valid_activation_function_derivatives():
  return activation_function_derivatives_mapping.keys()

def validate_activation_function_derivative(act_fn: str):
  return act_fn.lower() in get_valid_activation_function_derivatives()

def get_activation_function_derivative(act_fn: str):
  if validate_activation_function_derivative(act_fn):
    return activation_function_derivatives_mapping[act_fn.lower()]
  else:
    raise ValueError('{} is not a supported activation function with derivatives.', act_fn)