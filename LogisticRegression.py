import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method.
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __softmax(self, z):
        s = np.zeros(np.shape(z))
        for i in xrange(0,np.size(z,0)):
            denom = np.sum(np.exp(z[i]))
            s[i] = np.exp(z[i])/denom
        self.S = s
        return s

    def convertC(self, Y):
        C = np.zeros((np.size(Y),3))
        for y in xrange(0,np.size(Y)):
            C[y,Y[y]] = 1
        #print "C is ", C[0], np.shape(C)
        return C

    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        newX = np.hstack((np.ones((np.size(X,0),1)), X))
        print "newX is ", newX[0]
        self.C = C
        newC = self.convertC(C)
        w = np.zeros((3,np.size(newX[0])))
        cond = True
        i=0
        #grad = np.ones((np.shape(C,1),np.shape(X,1)))
        while i<10000:
            y = self.__softmax(np.dot(newX,w.T))
            if i - math.floor(i / 1000)*1000 == 0:
                print "np.dot(X,w.T) is ", np.dot(newX,w.T)[0], np.shape(np.dot(newX,w.T))
                print "t is ", y[0], "shape of t is ", np.shape(y)
                #print "C-t is ", newC-y
            grad = np.dot((y-newC).T,newX)+ self.lambda_parameter*w
            #for i in xrange(0,np.size(X[0])):
            #    grad[i] = np.dot((C[i]-t[i]).T,X[i])
            if i - math.floor(i / 1000)*1000 == 0:
                print "grad is ", grad
            w = w - self.eta*grad
            if i - math.floor(i / 1000)*1000 == 0:
                print "w is ",w[0]
            error = np.linalg.norm(np.absolute(grad)) > self.lambda_parameter
            i = i+1
        self.W = w

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        newX = np.hstack((np.ones((np.size(X_to_predict,0),1)), X_to_predict))
        t = self.__softmax(np.dot(newX,(self.W).T))
        #print "t predicted is ", t[0], np.shape(t)
        return np.argmax(t, axis=1)

    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        #print "xx.shape[0] = " , xx.shape[0], "xx.shape[1] = ", xx.shape[1]
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
