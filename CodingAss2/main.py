from utils import plot_data, generate_data
import numpy as np
"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

def gradient(X, t, w, b):
    '''
    compute the gradient of the cross-entropy loss
    '''
    Z = np.dot(X, w) + b
    logistic_func = 1 / (1 + np.exp(-Z))
    dw = np.dot((logistic_func - t), X)
    db = np.sum(logistic_func - t)
    return dw, db

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    w = np.zeros(X.shape[1])
    b = 0
    Lambda = 0.001
    for i in range(100):
        dw, db = gradient(X, t, w, b)
        w = w - Lambda * dw
        b = b - Lambda * db

    return w, b

def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    Z = np.dot(X, w) + b
    predict_value = 1 / (1 + np.exp(-Z))
    t = np.zeros(X.shape[0])
    temp = 0
    for i in predict_value:
        if i >= 0.5:
            t[temp] = 1
        temp += 1
    return t

def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    # X_: Nsample x (d+1)
    X_ = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
    temp = np.dot(X_.T, X_)
    temp_inv = np.linalg.inv(temp)
    w = np.dot(np.dot(temp_inv, X_.T), t)
    b = w[-1]
    w = w[0:-1]
    return w, b

def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    X_ = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
    w = np.append(w, b)
    predict_value = np.dot(X_, w)
    t = np.zeros(X.shape[0])
    temp = 0
    for i in predict_value:
        if i >= 0.5:
            t[temp] = 1
        temp += 1
    return t

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    count = 0
    for i in range(len(t)):
        if t[i] == t_hat[i]:
            count += 1
    acc = count / len(t)
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Training accuracy of linear regression on Dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Training accuracy of logistic regression on Dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Training accuracy of linear regression on Dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Training accuracy of logistic regression on Dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')

main()
