#Goal : Teach the computer the function y = 2x + 1
# We give it data, and it must "learn" m (slope) and b (intercept)

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
for _ in range(epochs):
    m_deriv = 0
    b_deriv = 0
    N = len(X)

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

    m -= learning_rate * (m_deriv / N)
    b -= learning_rate * (b_deriv / N)

print(f"Learned: y = {m:.2f}x + {b:.2f}")