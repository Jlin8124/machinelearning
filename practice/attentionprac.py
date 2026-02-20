import numpy as np

# Assume Q, K, and V are already created matrices (e.g., our sentence)
# d_k is the size of the dimension (e.g., 64)
# Create the 2x2 matrices using nested lists
Q = np.array([[2, 0], [0, 2]])
K = np.array([[2, 0], [0, 2]])
V = np.array([[10, 0], [0, 10]])

# Define the scaling factor we used
d_k = 1

# Step 1: The Matching (Dot Product of Q and K transposed, then scaled)
raw_scores = np.dot(Q, K.T) / np.sqrt(d_k)

# Step 2: The Scoring (Softmax turns raw scores into percentages/weights)
# e.g., turning [2.0, 0.1, -1.0] into [0.80, 0.15, 0.05]
attention_weights = np.exp(raw_scores) / np.sum(np.exp(raw_scores), axis=1, keepdims=True)

# Step 3: The Result (Multiply the percentage weights by the Value matrix)
output = np.dot(attention_weights, V)