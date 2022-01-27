import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Disparidad2Distancia:
    def __init__(self):
        self.x = np.array((0.9911572265625, 0.8864306640625, 0.7731, 0.6696337890625, 0.595791015625, 0.538887355638587, 0.4798193359375, 0.4418603515625, 0.412236328125, 0.3819091796875))
        self.y = np.array((60, 70, 80, 90, 100, 110, 120, 130, 140, 150))
        self.pf = PolynomialFeatures(degree = 6)
        X = self.pf.fit_transform(self.x.reshape(-1,1))
        self.model = LinearRegression()
        self.model.fit(X, self.y) 

    def plot(self):
        fig = plt.figure()
        
        x1base = range(100, 0, -2)
        x1 = np.array(x1base)
        x1 = x1.astype(float)
        x1 = x1 / 100.0
        X1 = self.pf.fit_transform(x1.reshape(-1,1))
        dist = self.model.predict(X1)       
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(x1, dist, 'r')
        plt.show()
        
    def obtenerDistancia (self, disp):
        x1 = np.array((disp))
        X1 = self.pf.fit_transform(x1.reshape(-1,1))
        dist = self.model.predict(X1)
        return dist[0]