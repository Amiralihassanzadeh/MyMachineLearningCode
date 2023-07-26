import numpy as np

class adalineGD (object) : 

    def __init__(self , eta =0.1 , n_iter = 50 , random_state = 1):
        self.eta = eta 
        self.n_iter = n_iter
        self.random_state = random_state 

    def fit(self, X , y): 

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc= 0 , scale= 0.1 , sixe = 1 + X.shape[1])

        self.cost_ = [] 

        for i in range (self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y -output
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum/2.0
            self.cost_.append(cost)
        
        return self
    

    def net_input(self,X):
        return np.dot(X,self.W_[1:]) +self.w_[0]
    
    def activation(self , X):
        return X

    def predict(self, X):

        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)
