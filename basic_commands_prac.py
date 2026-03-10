import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

x = np.array([[1, 2], [3,4]])

print (x.ndim)
print (x.shape)

x = np.array([1, 2, 3 , 4, 5, 6])

x_reshape = x.reshape((2,3))

print(x_reshape)

print(x_reshape[0][0])

x_reshape[0][0] = 5
print(x_reshape)

print(x)

np.sqrt
np.random.normal()

# help(np.random.normal)

y = np.random.normal(size=50)

#print(y)

z = y + np.random.normal(loc=50, scale=1, size=50)

print(z)

print(np.corrcoef(z, y))

rng = np.random.default_rng(3)
y = rng.standard_normal(10)
print (np.mean(y))
print (y.mean())

fig , ax = subplots(nrows=2, ncols=3, figsize=(15, 5))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
#x = np.sort(rng.standard_normal(100))
#y = np.sort(rng.standard_normal(100))
#ax.plot(x, y, 'x');
#ax.scatter(x,y, marker='o')
ax[0,1].plot(x, y, 'o')
ax[1,2]. scatter(x, y, marker='+')
ax[0,1].set_xlabel("this is the x-axis")
#ax.set_ylabel("this is the y-axis")
#ax.set_title("Plot of X vs Y")
#fig.set_size_inches(12,3)
fig

fig.tight_layout()
plt.show()