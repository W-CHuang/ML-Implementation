import numpy as np
class GradientDescent():
    '''
    A class written to implement gradient descent algorithm
    Will be containing 3 gradient descent algorithm. Batch gradient descent(BG), StochasticGradientDescent(SGD), miniBatchGradientDescent(mBGD)
    '''
    def __init__(self,X,y,theta,*args):
        self.alpha=0.01
        self.maxIter=1000
        self.converged=False
        self.theta=theta
        self.X=X
        self.y=y
        self.MSE=0.0
        self.BatchSize=16 ##Habitually set to the exponential of two
        self.m=len(y) ##number of features
        self.seed=0 ##ensure reproducity
        self.miniBatches=[] ##use to hold batches with structure [(,)].
    def mseCalculator(self,x_cal,y_cal,theta):
        '''
        X:training samples; Should be one row one feature (n,m)
        y:target vector
        theta:parameter that minimizes function of J,a vector of (m,1)
        '''
        predictions=np.dot(x_cal,theta)
        MSE=1/(2*len(y_cal))*np.sum(np.square(np.subtract(predictions,y_cal)))
        return MSE
    def miniBatchesAttainer(self):
        np.random.seed(self.seed)
        ##shuffle X,y
        self.miniBatches=[]
        sfl=np.random.permutation(self.m)   #1D array for shuffle
        XShuffled=self.X[sfl,:]
        yShuffled=self.y[sfl,:]
        if self.BatchSize>self.m:
            self.miniBatches=[(XShuffled,yShuffled)]
        else:
            quo=self.m//self.BatchSize   ##integer
            leftover=self.m%self.BatchSize   ##leftover
            ##append complete Batches of BatchSizes
            for q in range(0,quo,self.BatchSize):
                self.miniBatches.append((XShuffled[q:q+self.BatchSize],yShuffled[q:q+self.BatchSize]))
            if leftover:
                self.miniBatches.append((XShuffled[quo:-1],yShuffled[quo:-1]))
    def BatchGradient(self):
        Iter=0
        features=self.X.shape[-1]
        thetaRecord=np.empty(features)  ##empty array,features are the numbers of theta
        mseRecord=np.zeros(1)
        self.converged=False
        theta=self.theta
        while not self.converged and Iter<self.maxIter:
            prediction=np.dot(self.X,theta)
            theta=theta-1/self.m*self.alpha*np.sum(np.dot(self.X.T,np.subtract(prediction,self.y)))
            #print (theta)
            thetaRecord=np.c_[thetaRecord,theta]
            mse=self.mseCalculator(self.X,self.y,theta)
            if mseRecord[-1]==mse:  ## not correct; should do convergence analysis
                print ('Converged at Iter {d}'.format(d=Iter))
                self.converged=True
            mseRecord=np.append(mseRecord,self.MSE)
            Iter+=1
        if not self.converged:print ('Stop calculation after {d} iterations'.format(d=self.maxIter))
        return theta,thetaRecord,mseRecord
    def StochasticGradientDescent(self):
        '''
        to do Stochastic Batch Gradient Descent, first shuffle the sample order (Randomly select one)
        '''
        mseRecord=np.zeros(1)
        Iter=0
        self.converged=False
        theta=self.theta
        mse=0
        while not self.converged and Iter<self.maxIter:
            for loop in range(self.m):
                i=np.random.randint(0,self.m)
                x_i=self.X[i,:].reshape(1,self.X.shape[1]) ##make sure only one sample (training example) is selected
                y_i=self.y[i].reshape(1,1) ##ensure that correct according y (target value) is selected
                theta=theta-1/self.m*self.alpha*np.sum(np.dot(x_i.T,np.subtract(np.dot(x_i,theta),y_i)))
                mse+=self.mseCalculator(x_i,y_i,theta)
            if mse==mseRecord[-1]:      ##not correct; should do convergence analysis
                self.converged=True
                print ('Converged at Iter {d}'.format(d=Iter))
            mseRecord=np.append(mseRecord,mse)
            Iter+=1
        if not self.converged:
            print ('Stop calculation after {d} iterations'.format(d=self.maxIter))
        return theta,mseRecord
    def miniBatchGradientDescent(self):
        '''
        Get Batches at the first step and starting trainning calculating the cost
        '''
        mseRecord=np.empty(1)
        Iter=0
        mse=0
        theta=self.theta
        self.miniBatchesAttainer()
        BacthNum=len(self.miniBatches)
        while Iter<self.maxIter:
            for loop in range(BacthNum):
                randBacthes=np.random.randint(0,BacthNum)
                x_i,y_i=self.miniBatches[randBacthes]
                theta=theta-1/len(y_i)*self.alpha*np.sum(np.dot(x_i.T,np.subtract(np.dot(x_i,theta),y_i)))
                mse+=self.mseCalculator(x_i,y_i,theta)
            mseRecord=np.append(mseRecord,mse)
            Iter+=1
        return theta,mseRecord
    def normalEquation(self):
        '''
        Matrix Operation for calculating theta

        '''
        theta=np.dot(np.dot(np.linalg.inv(np.dot(self.X.T,self.X)),self.X.T),self.y)
        mseRecord=self.mseCalculator(self.X,self.y,theta)
        return theta,mseRecord
x = 2*np.random.rand(100,1)
y = 3+3* x+np.random.randn(100,1)
theta=np.random.randn(2,1)
X_b =np.c_[np.ones((len(x),1)),x]
G=GradientDescent(X_b,y,theta)
theta,thetaRecord,mseRecord=G.BatchGradient()
theta2,mseRecord2=G.StochasticGradientDescent()
theta3,mseRecord3=G.miniBatchGradientDescent()
theta4,mseRecord4=G.normalEquation()
print (theta)
print (theta2)
print (theta3)
print (theta4)
print (mseRecord[-1])
print (mseRecord2[-1]) 