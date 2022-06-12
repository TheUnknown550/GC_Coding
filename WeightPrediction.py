from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.array ([10121,20121,30121,40121,50121,60121,70121,80121,90121,100121,110121,120121,130121,140121,150121,160121,170121,180121,190121,200121,210121,220121,230121,240121,250121,260121,270121,280121,290121,300121,310121,10221,20221])
y = []
for i in range(len(x)):
    print('Please enter your weight for ', x[i])
    inp = float(input(': '))
    y.append(inp)

x = x.reshape(-1, 1)

plt.scatter(x, y, s = 80, marker='+')

LR = LinearRegression(normalize=True)
LR.fit(x, y)
y_pred = LR.predict(x)

rmse = np.sqrt(mean_squared_error(y, y_pred))
print ('RMSE (Root Mean Squared Error) = ', rmse)

r2 = r2_score(y, y_pred)
print ('R2 = ', r2)

# Plot ข้อมูล (x,y) 
plt.scatter(x, y, s = 80, marker = '+', label = 'Data')

# Plot Linear Regression Model (x,y_pred)
plt.plot(x, y_pred, linewidth = 4, color = 'r', label = 'Linear Regression Model')

date = int(input("Enter Date: "))
print(LR.predict([[date]]))

plt.legend(loc='lower right')
plt.show()