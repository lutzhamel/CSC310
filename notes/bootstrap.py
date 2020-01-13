import pandas as pd
from numpy import percentile
from sklearn.model_selection import train_test_split

def bootstrap(model,D,target_name):
    rows = D.shape[0]
    acc_list = []
    for i in range(200):
        B = D.sample(n=rows,replace=True)
        X = B.drop(target_name,1)
        y = B[target_name]
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2)
        model.fit(train_X, train_y)
        acc_list.append(model.score(test_X, test_y))
    acc_list.sort()
    ub = percentile(acc_list,97.5)
    lb = percentile(acc_list,2.5)
    return (lb, ub)

if __name__ == '__main__':
    from sklearn import tree

    # classification
    t1c = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    t2c = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)

    print("******** abalone ***********")
    df = pd.read_csv("assets/abalone.csv")
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1c,df,'sex')))
    print("Confidence interval max_depth=None: {}".format(bootstrap(t2c,df,'sex')))

    # regression
    t1r = tree.DecisionTreeRegressor(max_depth=3)
    t2r = tree.DecisionTreeRegressor(max_depth=None)

    print("******** cars ***********")
    df = pd.read_csv("assets/cars.csv")
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1r,df,'dist')))
    print("Confidence interval max_depth=None: {}".format(bootstrap(t2r,df,'dist')))

