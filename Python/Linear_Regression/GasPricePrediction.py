import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.array ([50165,80165,110165,140165,200165,250165,280165,20265,80265,120265,160265,180265,190265,240265,260265,10365,20365,40365,50365,80365,90365,100365,150365,170365,220365,230365,260365,290365,310365,20465,20465,90465,190465,230465,270465,290465,10565,40565,60565,70565,120565,160565,240565,280565,310565,10665,70665,80665])
y = np.array ([29.04,29.44,29.84,29.84,29.94,29.94,29.94,29.94,29.94,29.94,29.94,27.94,27.94,28.54,29.14,29.14,29.74,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,31.94,31.94,31.94,31.94,31.94,31.94,31.94,32.94,32.94,33.94,33.94])
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


print(LR.predict([[200565]]))

plt.legend(loc='lower right')
plt.show()