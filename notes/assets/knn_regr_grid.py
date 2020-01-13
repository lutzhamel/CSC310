# Import the necessary modules and libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16)) # noise every 5th point

# setting up grid search
model = KNeighborsRegressor()
param_grid = {'n_neighbors': list(range(1,10))}
grid = GridSearchCV(model, param_grid, cv=3)

# performing grid search
grid.fit(X,y)

# print out what we found
print("Best parameters: {}".format(grid.best_params_))

# Predict
regr_1 = grid.best_estimator_
print("R^2 score: {}".format(regr_1.score(X,y)))
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, c="darkorange")
plt.plot(X_test, y_1, color="cornflowerblue", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("KNN Regression using Gridsearch")
plt.show()

