"""
Docstring for ltsm_networks.ltsm_network

Raw implementation of the basic math for LTSM Networks.
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# 2. Starting Data (Represented as column vectors)
x_t = np.array([[1.2], 
                [0.8]])     # Current input (size 2x1)

h_prev = np.array([[0.5], 
                   [-0.1]]) # Previous hidden state (size 2x1)

C_prev = np.array([[1.0], 
                   [0.5]])  # Previous cell state (size 2x1)

stacked_size = 4
hidden_size = 2

# Forget weights and bias
W_f = np.random.randn(hidden_size, stacked_size)
b_f = np.zeros((hidden_size, 1))

# 2. Input Gate weights and bias
W_i = np.random.randn(hidden_size, stacked_size)
b_i = np.zeros((hidden_size, 1))

# 3. Candidate Memory weights and bias
W_c = np.random.randn(hidden_size, stacked_size)
b_c = np.zeros((hidden_size, 1))

# 4. Output Gate weights and bias
W_o = np.random.randn(hidden_size, stacked_size)
b_o = np.zeros((hidden_size, 1))

# 1. Concatenate the previous hidden state and current input
stacked_input = np.vstack((h_prev, x_t))

# 2. Forget Gate Calculation
f_t = sigmoid(np.dot(W_f, stacked_input) + b_f)

# 3. Input Gate & Candidate Memory Calculation
i_t = sigmoid(np.dot(W_i, stacked_input) + b_i)
C_tilde = tanh(np.dot(W_c, stacked_input) + b_c)

# 4. Update the Cell State (The "Conveyor Belt")
# Note: * is element-wise multiplication in NumPy
C_t = f_t * C_prev + i_t * C_tilde

# 5. Output Gate & Hidden State Calculation
o_t = sigmoid(np.dot(W_o, stacked_input) + b_o)
h_t = o_t * tanh(C_t)

print("Updated Cell State (Long-term Memory):\n", C_t)
print("Updated Hidden State (Short-term Output):\n", h_t)