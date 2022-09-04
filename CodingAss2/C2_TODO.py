# -*- coding: utf-8 -*-

import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

def readMNISTdata():
    with open('t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels

def softmax(z):
    """
    the softmax function which can avoid the overflow
    """
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp

def predict(X, w, b, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    # TODO Your code here
    y = np.dot(X, w)
    z = softmax(y)

    t_hat = np.argmax(z, axis=1)
    loss = -np.sum(t * np.log(z[np.arange(len(t))])) / len(t)
    count = 0
    for i in range(len(t)):
        if t_hat[i] == t[i]:
            count += 1
    acc = count / len(t)
    return y, t_hat, loss, acc



def train(X_train, y_train, X_val, t_val, X_test, t_test):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    #TODO Your code here
    w = np.random.random((X_train.shape[1], 10))
    b = np.random.random(10)

    losses_train = []
    val_result = []
    test_result = []
    acc_best = -1000
    test_best = -1000
    epoch = 0
    W_best = None
    b_best = None

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train / batch_size)) ):
            X_batch = X_train[b * batch_size : (b+1) * batch_size]
            y_batch = y_train[b * batch_size : (b+1) * batch_size]

            y = np.dot(X_batch, w) + b
            z = softmax(y)
            y_hat = np.zeros((len(y_batch), 10))
            for i in range(len(y)):
                y_hat[i, y_batch[i]] = 1

            dw = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (z - y_hat))
            db = (1 / X_batch.shape[0]) * np.sum(z - y_hat)
            w = w - alpha * dw
            b = b - alpha * db
            loss_batch = -np.sum(y_batch * np.log(z[np.arange(len(y_batch))])) / len(y_batch)
            loss_this_epoch += loss_batch

        # monitor model behavior after each epoch
        # Compute the training loss by averaging loss_this_epoch
        size = int(np.ceil(N_train/batch_size))
        losses_train.append(loss_this_epoch / size)

        # Perform validation on the validation test 
        _, _, _, val_acc = predict(X_val, w, b, t_val)
        val_result.append(val_acc)

        _, _, _, test_acc = predict(X_test, w, b, t_test)
        test_result.append(test_acc)

        # Keep track of the best validation epoch, risk, and the weights
        if val_acc > acc_best:
            acc_best = val_acc
            test_best = test_acc
            epoch_best = epoch
            W_best = w
            b_best = b
    return epoch_best, acc_best,  W_best, b_best,  losses_train, val_result, test_result, test_best


##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()
X_test = X_test / 256
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

# setting
N_class = 10
alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# Report numbers and draw plots as required. 
epoch_best, acc_best, W_best, b_best, loss_train, val_acc, test_acc, test_best = train(X_train, t_train, X_val, t_val, X_test, t_test)
print("The number of epoch that yields the best validation performance is:", epoch_best)
print("The validation performance (accuracy) in this epoch is:", acc_best)
print("The test performance (accuracy) in this epoch is:", test_best)

# plot of training loss 
plt.figure()
plt.plot(np.arange(1, 51), loss_train)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch")
plt.savefig("figure1.jpg")
plt.show()

# plot of validation accuracy
plt.figure()
plt.plot(np.arange(1, 51), val_acc)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Epoch")
plt.savefig("figure2.jpg")
plt.show()

