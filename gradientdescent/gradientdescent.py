#Goal : Teach the computer the function y = 2x + 1
# We give it data, and it must "learn" m (slope) and b (intercept)

import matplotlib.pyplot as plt

#1. Data

X = [1, 2, 3, 4, 5]
Y = [3, 5, 7, 9, 11] # Actual relationship: 2x + 1

#2 Random Intialization

m = 0.0
b = 0.0

#3 Hyperparameters

learning_rate = 0.01 # How big of a step to take
epochs = 1000

#4 Training Loop
loss_history = []

for _ in range(epochs):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    epoch_loss = 0

    for i in range(N):
        # Forward Pass: Make a prediction
        y_pred = m * X[i] + b

        # Error: How far off were we?
        error = y_pred - Y[i]

        #J = (ypred - yactual) ** 2
        #In this case ypred = m * X[i] + b
        # J = (m * X[i] + b - yactual) ** 2
        # Derivative of J or the minimum cost function to find the slope
        # J' = 2*(m * X[i] + b)
        m_deriv += 2 * error * X[i]
        b_deriv += 2 * error
        epoch_loss += error ** 2
    m -= learning_rate * (m_deriv / N)
    b -= learning_rate * (b_deriv / N)
    loss_history.append(epoch_loss / N) # MSE for this epoch
print(f"Learned: y = {m:.2f}x + {b:.2f}")

#5 Predictions on new data
print("\nPredictions:")
for x_new in [6, 7, 10]:
    print(f"  x={x_new} → y_pred={m * x_new + b:.2f}  (expected {2 * x_new + 1})")

#6 Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Loss curve
ax1.plot(loss_history)
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")

# Plot 2: Learned line vs data points
ax2.scatter(X, Y, color="blue", label="Training data")
x_line = [0, 6]
y_line = [m * x + b for x in x_line]
ax2.plot(x_line, y_line, color="red", label=f"Learned: y={m:.2f}x+{b:.2f}")
ax2.set_title("Learned Line vs Data")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()

plt.tight_layout()
plt.savefig("gradientdescent_results.png")
plt.show()
print("Plot saved to gradientdescent_results.png")