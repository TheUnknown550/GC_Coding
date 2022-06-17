import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.array ([10121,20121,30121,40121,50121,60121,70121,80121,90121,100121,110121,120121,130121,140121,150121,160121,170121,180121,190121,200121,210121,220121,230121,240121,250121,260121,270121,280121,290121,300121,310121,10221,20221])
y = []

for i in range(len(x)):
    print('Please enter your weight for ', x[i])
    inp = float(input(': '))
    y.append(inp)


x = x.reshape(-1, 1)

poly_feature = PolynomialFeatures(degree=3) 
x_poly = poly_feature.fit_transform(x)

poly_model = LinearRegression()
poly_model = poly_model.fit(x_poly, y)
y_pred_poly = poly_model.predict(x_poly)

rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
print ('RMSE (Root Mean Squared Error) = ', rmse_poly)

r2_poly = r2_score(y, y_pred_poly)
print ('R2 = ', r2_poly)

date = int(input("Enter Epoch Date: "))
data_x = np.array ([date]) #เลขที่ทดสอบ
data_x = data_x.reshape(-1, 1)
pred_data_poly = poly_feature.fit_transform(data_x)

pred_data_poly_x = poly_model.predict(pred_data_poly)
print(pred_data_poly_x)

# Plot ข้อมูล (x,y) 
plt.scatter(x,y, s = 80, marker='+', label='Data')

# เตรียมข้อมูล และ Plot Polynomial Regression Model (x,y_pred_poly) 
sorted_zip = sorted(zip(x, y_pred_poly))
x_plot, y_pred_plot = zip(*sorted_zip)
plt.plot(x_plot, y_pred_plot, linewidth=4, color='r', label='Polynomial')

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
# plt.axis([27, 62, 1500, 4600])
plt.legend(loc = 'lower right')
plt.show()