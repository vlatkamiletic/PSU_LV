import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy

x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

#stupnjevi dodatnih velicina
degrees = [2, 6, 15]

MSE_train = []
MSE_test = []

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree=degree)
    xnew = poly.fit_transform(x)
    
    np.random.seed(12)
    indeksi = np.random.permutation(len(xnew))
    indeksi_train = indeksi[:int(np.floor(0.7*len(xnew)))]
    indeksi_test = indeksi[int(np.floor(0.7*len(xnew)))+1:]
    
    xtrain = xnew[indeksi_train,]
    ytrain = y_measured[indeksi_train]

    xtest = xnew[indeksi_test,]
    ytest = y_measured[indeksi_test]
    
    #treniranje
    linearModel = lm.LinearRegression()
    linearModel.fit(xtrain,ytrain)
    
    #Predikcija na skupu za učenje
    ytrain_p = linearModel.predict(xtrain)
    # MSE
    MSE_train.append(mean_squared_error(ytrain, ytrain_p))
    
    #Predikcija na skupu za testiranje
    ytest_p = linearModel.predict(xtest)
    # MSE
    MSE_test.append(mean_squared_error(ytest, ytest_p))
    
    plt.subplot(1, len(degrees), i+1)
    plt.plot(x, y_true, label='f')
    plt.plot(xtest[:,1], ytest_p, 'og', label='predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Degree {degree}')
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

print(f"MSE for training data: {MSE_train}")
print(f"MSE for test data: {MSE_test}")