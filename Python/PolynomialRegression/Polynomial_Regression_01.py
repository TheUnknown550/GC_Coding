import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.array ([1654646400,1654560000,1654059600,1669870800,1669611600,1669266000,1668574800,1668574800,1668229200,1667797200,1667710800,1667538000,1667278800,1651208400,1651035600,1650690000,1650344400,1649480400,1649394000,1649134800,1648875600,1648702800,1648530000,1648270800,1648011600,1647925200,1647493200,1647320400,1646888400,1646802000,1646715600,1646456400,1646370000,1646197200,1646110800,1645851600,1645678800,1645246800,1645160400,1644987600,1644642000,1644296400,1643778000,1643346000,1643086800,1642654800,1642136400,1641877200,1641618000,1641358800])
y = np.array ([33.94,33.94,32.94,32.94,31.94,31.94,31.94,31.94,31.94,31.94,31.94,31.94,31.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.74,29.14,29.14,28.54,27.94,27.94,29.94,29.94,29.94,29.94,29.94,29.94,29.94,29.84,29.84,29.44,29.04])

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