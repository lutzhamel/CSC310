import pandas as pd
from numpy import percentile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def bootstrap(model,D,target_name):
    rows,__ = D.shape
    acc_list = []
    for i in range(200):
        B = D.sample(n=rows,replace=True)
        X = B.drop(target_name,1)
        y = B[target_name]
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
        model.fit(train_X, train_y)
        predict_y = model.predict(test_X)
        acc_list.append(accuracy_score(test_y, predict_y))
    acc_list.sort()
    ub = percentile(acc_list,97.5)
    lb = percentile(acc_list,2.5)
    return (lb, ub)

if __name__ == '__main__':
    from sklearn import tree

    t1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    t2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)

    print("******** iris ***********")
    df = pd.read_csv("assets/iris.csv")
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1,df,'Species')))
    print("Confidence interval max_depth=None: {}".format(bootstrap(t2,df,'Species')))

    print("******** abalone ***********")
    df = pd.read_csv("assets/abalone.csv")
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1,df,'sex')))
    print("Confidence interval max_depth=None: {}".format(bootstrap(t2,df,'sex')))
