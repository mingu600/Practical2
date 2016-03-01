from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method.
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    def __convert(self, Y):
        self.K = np.max(Y)+1
        print "convert function ", np.max(Y)
        C = np.zeros((np.size(Y),self.K))
        for y in xrange(0,np.size(Y)):
            C[y,Y[y]] = 1
        print "C is ", C, np.shape(C)
        return C

    def __findPi(self,Y):
        total = np.size(Y,0)
        self.prior = self.ysum/total
        print "prior pi is ", self.prior

    def __classCondProb(self, n1,n2,n3):
        self.condProb = np.vstack((n1,n2,n3)).T
        print "condProb is ", self.condProb[0], np.shape(self.condProb)

    def __covariance(self,X,mu,Y,c):
        cons = 1/self.ysum[c]
        print "constant for cov ", cons
        cov = np.dot((Y[:,c]*(X - mu[c]).T), (X - mu[c]))
        print "cov matrix Sigma is ", cons*cov
        return cons*cov

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = np.size(X,0)
        self.D = np.size(X,1)
        newY = self.__convert(Y)
        print "new Y is ", newY
        self.ysum = np.sum(newY,0)
        print "ysum is ", self.ysum
        #find pi
        self.__findPi(newY)
        #find mu
        self.mu = (np.dot(newY.T,X).T/self.ysum).T
        print "mu is ", self.mu
        #find sigma for shared and not shared
        self.S1 = self.__covariance(X,self.mu,newY,0)
        self.S2 = self.__covariance(X,self.mu,newY,1)
        self.S3 = self.__covariance(X,self.mu,newY,2)
        self.S = self.ysum[0]*self.S1/np.sum(self.ysum) + self.ysum[1]*self.S2/np.sum(self.ysum) + self.ysum[2]*self.S3/np.sum(self.ysum)
        print "dependent cov matrix ", self.S
        if self.isSharedCovariance:
            self.S1 = self.S
            self.S2 = self.S
            self.S3 = self.S
        #if isSharedCovariance:
        #    prob = self.__gaussianPdf(X,mu,S)
        #    self.__classCondProb(prob,prob,prob)
        #else:
        #    prob1 = self.__gaussianPdf(X,mu,S1)
        #    prob2 = self.__gaussianPdf(X,mu,S2)
        #    prob3 = self.__gaussianPdf(X,mu,S3)
        #    self.__classCondProb(prob1,prob2,prob3)
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        #if self.isSharedCovariance:
        #    prob = self.__gaussianPdf(X_to_predict,self.mu,self.S)
        #    self.__classCondProb(prob,prob,prob)
        #else:
        prob1 = multivariate_normal.pdf(X_to_predict,self.mu[0],self.S1)
        prob2 = multivariate_normal.pdf(X_to_predict,self.mu[1],self.S2)
        prob3 = multivariate_normal.pdf(X_to_predict,self.mu[2],self.S3)
        self.__classCondProb(prob1,prob2,prob3)
        print "condprob is ", self.condProb[0]
        Y = self.condProb*self.prior
        print "output Y is ", Y[0], np.shape(Y)
        return np.argmax(Y, axis = 1)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
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
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
