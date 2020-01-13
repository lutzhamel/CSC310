# generate the data
import matplotlib.pyplot as plt
import random
import pandas
depth  = int(input('Choose a number for tree depth: '))

x = pandas.DataFrame([10 * random.random() for __ in range(50)])
y = 2 * x - 1 + pandas.DataFrame([random.random() for __ in range(50)])

# pick model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=depth)
model.fit(x, y)

# plot the model together with the data
xfit = pandas.DataFrame([i for i in range(-1, 12)])
yfit = model.predict(xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

# compute the R^2 score
print("R^2 score: {}".format(model.score(x,y)))


