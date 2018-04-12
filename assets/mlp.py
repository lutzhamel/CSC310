import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# get data
df = pd.read_csv("clean_biopsy.csv", na_values='NA')
df = df.drop(['IX','ID'],axis=1)
X  = df.drop(['class'],axis=1)
y = df['class']


# neural network
model = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000)

# do the 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5)
print("Fold Accuracies: {}".format(scores))
print("Accuracy: {}".format(scores.mean()))
