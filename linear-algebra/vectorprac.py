import numpy as np

u = np.array([1,0,0])
v = np.array([0,1,0])
w = np.array([0,0,1])

center = 1/3*u + 1/3*v + 1/3*w
print("center:", center)

corner = 1*u + 0*v + 0*w
print("corner:", corner)

# Midpoint of edge between u and v
midpoint = 0.5*u + 0.5*v + 0*w
print("midpoint:", midpoint)

# Verify they sum to 1
c, d, e = 1/3, 1/3, 1/3
print("sum of coefficients:", c+d+e)