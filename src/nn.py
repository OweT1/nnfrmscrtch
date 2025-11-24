import numpy as np
from typing import Literal

from src.activation_functions import (
  get_valid_activation_functions,
  validate_activation_function,
  get_activation_function
)
from src.helper import (
  get_initial_params,
  one_hot_encode
)

class NeuralNetwork:
  def __init__(self, input_size: int, activation_fn: str):
    self.layers = [(input_size, activation_fn, 0)]
    self.layer_params = {}
  
  def _get_last_layer_num(self):
    return self.layers[-1][-1]
  
  def _init_layer_params(self):
    if len(self.layers) > 1 and not self.layer_params:
      for i in range(len(self.layers)-1):
        layer_c = self.layers[i]
        layer_n = self.layers[i+1]
        layer_size_c, _, layer_num_c = layer_c
        layer_size_n, _, _ = layer_n
        W = get_inital_params(layer_size_n, layer_size_c)
        b = get_initial_params(layer_size_n, 1)
        self.layer_params[f"W{i}"] = W
        self.layer_params[f"b{i}"] = b
    else:
      print("layer params have already been initialised")
  
  def add_layer(self, layer_size: int, activation_fn: str):
    if validate_activation_function(activation_fn):
      layer_num = self._get_last_layer_num() + 1
      self.layers.append((layer_size, activation_fn, layer_num))
    else:
      raise ValueError('{} is not a supported activation function. Please use any of: {}', activation_fn, get_valid_activation_functions())
  
  def _forward_one(self, W, b, X: np.ndarray, act_fn):
    Z = W.dot(X) + b # Z1 will be n samples x layer_size dim
    A = act_fn(Z)
    return Z, A
  
  def forward(self, X: np.ndarray):
    weights = {}
    for i in range(len(self.layers)-1):
      layer = self.layers[i]
      _, act_function, layer_num = layer
      layer_params_w, layer_params_b = self.layer_params[f'W{layer_num}'], self.layer_params[f'b{layer_num}']
      act_fn = get_activation_function(act_function)
      Z, A = _forward_one(layer_params_w, layer_params_b, X, act_fn)
      weights[f'Z{layer_num}'] = Z
      weights[f'A{layer_num}'] = A
      X = A
    return weights 
  
  def backward(self, weights, X: np.ndarray, Y: np.ndarray):
    one_hot_Y = one_hot_encode(Y)
    m = Y.size
    
  def params_update(self, weight_changes, learning_rate=0.05):
    for layer, weights in self.layer_params.items():
      weight_change = weight_changes[f"d{layer}"]
      self.layer_params[layer] = weights - learning_rate * weight_change
  
  def train(self, input_X, input_Y, iterations: int, learning_rate=0.05):
    self._init_layer_params() # Initialise the network parameters
    for i in range(iterations):
      weights = self.forward(input_X)
      weight_changes = self.backward(input_X, input_Y, weights)
      self.params_update(weight_changes, learning_rate)
      
  def predict(self, X):
    last_layer_num = self._get_last_layer_num()
    weights = self.forward(X)
    output_weights = weights[f'A{last_layer_num}']
    