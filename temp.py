import numpy as np
import matplotlib.pyplot as plt

# Example data
data = np.random.rand(10, 10)

# Create heatmap
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()  # Add color bar indicating the scale
plt.show()
