from sklearn.svm import SVC
import joblib

class  SVMClassifier:
    def __init__(self, kernel="linear", c=0.1):
        self.model = SVC(
            kernel=kernel,
            C=c,
            probability=True
        )        
    def train(self, X, y):
        """
            train the model with SVM classifier 
        """
        self.model.fit(X=X, y=y)
    def predict(self, X, proba=False):
        """
        return predict the output 'Class' or probabilty
        """
        if proba:
           return self._predict_proba(X)
        
        return self._predict(X)

    
    def _predict(self, X):
        """
        predict new sample 
        """
        return self.model.predict(X)
    
    def _predict_proba(self, X):
        """
        return probability of class 

        """
        return self.model.predict_proba(X)
    
    def save(self, path):
            joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


