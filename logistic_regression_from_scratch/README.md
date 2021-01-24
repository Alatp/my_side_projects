This is my attempt at coding logistic regression from scratch

class LogisticRegression():
    def __init__(self, lr=0.001, n_iter=500):
        self.lr = lr 
        self.n_iter = n_iter
        self.weights = None
        self.bias = None 
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear_model = self.bias + np.dot(x, self.weights) 
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.lr * dw 
            self.bias = self.bias - self.bias * db
    
    def predict(self, x_test):
        linear_model = self.bias + np.dot(x_test, self.weights) 
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls 
