"""Softmax."""



'''
If we multiply the scores by 10, the softmax goes to 1 and 0s.
Similarly, if we divide the scores by 10, the softmax goes to 
uniform.
'''
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis = 0)

# The scores should sum up to 1
print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

# Create an array from -2.0 to 5.9 with 0.1 interval
x = np.arange(-2.0, 6.0, 0.1)

# Stack the array vertically with 1s and 0.2s
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

'''
Compare the scores among the three arrays,
i.e., [-2.0, 1.0, 0.2], [-1.9, 1.0, 0.2]
The higher values go towards 1 and lower values go towards 0
'''
plt.plot(x, softmax(scores).T, linewidth=2)	 # .T = transpose
plt.ylabel('Scores')
plt.xlabel('Values')
plt.title('Softmax Score Example')
plt.legend(['x','1.0','0.2'])
plt.show()