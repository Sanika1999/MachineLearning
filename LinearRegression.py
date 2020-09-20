
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import datasets
dataset=pd.read_csv('Salary_Data (1).csv')
x=dataset.iloc[:,:-1].values#selects all columns except last
y=dataset.iloc[:,-1].values#selects dependent variable
#splitting dataset in training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#training the model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression() #regressor is an instance of class linear regression
regressor.fit(x_train,y_train)

#predict test set results
y_pred=regressor.predict(x_test)

#visualising training set
plt.scatter(x_train,y_train,color='violet')
plt.plot(x_train,regressor.predict(x_train), color='yellow')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()



#visualising testing set
plt.scatter(x_test,y_test,color='violet')
plt.plot(x_test,y_pred,color='yellow')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()
