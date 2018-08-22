# import matplotlib,numpy and sklearn
import matplotlib.pyplot as plt
import numpy as np

#importing model from scikit-learn
from sklearn import linear_model

#loading the datasets
house_prices=[245,312,279,308,199,219,405,324,319,255]
house_size=[1400,1600,1700,1875,1100,1550,2350,2450,1425,1700]
print(house_size)
#reshaping data to input
new_house_size=np.array(house_size).reshape((-1,1))
print(new_house_size)

#load model
reg_model=linear_model.LinearRegression()

#fit data into model
reg_model.fit(new_house_size,house_prices)

#coeffient
print("Coefficient :",reg_model.coef_)
#intercept
print("Intercept : ",reg_model.intercept_)

def graph(formula,x_range):
    x=np.array(x_range)
    y=eval(formula)
    plt.plot(x,y)

#plotting the predictions
graph('reg_model.coef_*x+reg_model.intercept_',range(1000,2700))
plt.scatter(new_house_size,house_prices,color='black')
plt.ylabel("House Prices")
plt.xlabel("House sizes")
plt.show()



