import csv # will help in importing the csv files
import numpy as np # will help in performing calculations
from sklearn.svm import SVR # help us building predictive models
import matplotlib.pyplot as plt # help plotting the data


dates = []  
prices = []


plt.switch_backend('TkAgg')

# this function will fill these arrays with relevent data
def get_data(filename):
	with open(filename, 'r') as csvfile: # 'r' parameter means that the file will be opened in read only mode
		csvFileReader = csv.reader(csvfile) # this is a file reader variable which will help to iterate over every row in csv file
		next(csvFileReader) # next is called here since it will skip the first row which is the column name
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[2])) # adding the date
			prices.append(float(row[1])) # adding the price for the particular date
	return


#function to create prediective price models and graph them
def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svr_lin = SVR(kernel='linear', C=1e3) 
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	# fitting all the data models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	svr_rbf.fit(dates, prices)


	#plotting the data
	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
	plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
	plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('AAPL.csv')

predicted_price = predict_prices(dates, prices, 29)

print(predicted_price)