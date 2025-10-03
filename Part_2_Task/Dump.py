
# data_acquisition = DataAcquisition("../Task_3_4_5/Exports_goods.csv")
# data_acquisition.acquire_data()
# x= data_acquisition.values_2024
# y= data_acquisition.values_2025
# # How do we know what to put in Y 
# X=x.reshape(-1,1)
# model = LinearRegression()
# model.fit(X,y)
# y_pred = model.predict(X)

# # Plot
# plt.figure(figsize=(8,6))
# plt.scatter(X, y, color='blue', label='real data')
# plt.plot(X, y_pred, color='red', linewidth=2, label='linear regression')
# plt.xlabel('Value (NOK million) - July 2024')
# plt.ylabel('Value (NOK million) - July 2025')
# plt.title('Régression linéaire entre 2024 et 2025')
# plt.legend()
# plt.grid(True)
# plt.show()