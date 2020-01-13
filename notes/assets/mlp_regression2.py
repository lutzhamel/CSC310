
n  = int(input('Choose a number for hidden nodes: '))

# Import the necessary modules and libraries
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
#regr_1 = MLPRegressor(hidden_layer_sizes=(n,), activation='logistic', max_iter=10000)
regr_1 = MLPRegressor(hidden_layer_sizes=(n,n),  max_iter=10000)
regr_1.fit(X, y)
print("R^2 score: {}".format(regr_1.score(X,y)))

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="hidden nodes="+str(n), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("MLP Regression")
plt.legend()
plt.show()
