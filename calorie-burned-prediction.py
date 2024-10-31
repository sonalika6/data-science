# -*- coding: utf-8 -*-
"""
Spyder Editor
.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from scipy.stats import pearsonr
from pandas import read_csv
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\exercise.csv") #enter your file path
#last 10 entries
last_x_rows = df.tail(10)
print("last 10 entries:",last_x_rows)

#first 10 entries
first_x_rows=df.head(10)
print("first 10 entries:",first_x_rows)

# Assuming the dataset is a CSV file named "swiggy.csv" in your Downloads folder

#plot scatter plot
# Create a scatter plot (assuming columns named 'Body_Temp' and 'Heart_Rates')
colors = ['blue', 'green'] #color of the dots
plt.scatter(df['Heart_Rate'], df['Body_Temp'])

# Add labels and title
plt.xlabel('Body_Temp')
plt.ylabel('Heart_Rate')
plt.title('scatter plot')
plt.show()

#histogram 
plt.hist(df['Duration'])
plt.title('Histogram')

#box plot
df.boxplot(by='Body_Temp', column="Heart_Rate", grid=False)
plt.title('box plot')
plt.title('box plot')
# Show the plot
plt.show()

#person correlation
list1=df["Body_Temp"]
list2=df["Heart_Rate"]

corr,_=pearsonr(list1,list2)
print("person correlation is:",corr)

#independent variable=Body_Temp
#dependant variable=Heart_Rates


#linear relationship
X = df[['Body_Temp']].values.reshape(-1,1) # Independent variable (Body_Temp)
y = df['Heart_Rate'].values # Dependent variable (Heart_Rates)

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict ratings based on the Body_Temp
predicted_ratings = model.predict(X)

# Plot the original data
plt.scatter(df['Body_Temp'], df['Heart_Rate'], color='blue', label='Actual Ratings')

# Plot the regression line (Predicted ratings)
plt.plot(df['Body_Temp'], predicted_ratings, color='red', label='Predicted Ratings')

plt.xlabel('Body_Temp')
plt.ylabel('Heart_Rate')
plt.title('Body_Temp vs Heart_Rate with Regression Line')
plt.legend()
plt.show()

# Print the model's slope and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
