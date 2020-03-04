# compute 95% confidence intervals for classification and regression
# problems

def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    from statsmodels.stats.proportion import proportion_confint
    alpha = 1 - .95 # 95% confidence interval
    lb, ub = proportion_confint(100*acc, n, alpha)
    return (lb,ub)

def regression_confint(rs_score, n, k):
    '''
    Compute the 95% confidence interval for a regression problem.
      rs_score -- R^2 score
      n        -- number of observations used to compute the R^2 score
      k        -- number of independent variables in dataset
    Returns a tuple (lb,ub)

    Reference:
    https://books.google.com/books?id=gkalyqTMXNEC&pg=PA88#v=onepage&q&f=false
    '''
    import math
    interval = 2*math.sqrt((4*rs_score*(1-rs_score)**2*(n-k-1)**2)/((n**2 - 1)*(n+3)))
    lb = rs_score - interval
    ub = rs_score + interval
    return (lb,ub)
