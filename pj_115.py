from turtle import color
import pandas as pd
import plotly.express as px


df = pd.read_csv("escape_velocity.csv")
velocity_list = df ["Velocity"].tolist()
escaped_list = df ["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()


import numpy as np
velocity_array = np.array(velocity_list)
escaped_array = np.array(escaped_list)

#Slope and intercept using pre-built function of Numpy
m, c = np.polyfit(velocity_array, escaped_array, 1)

y = []
for x in velocity_list:
  y_value = m*x + c
  y.append(y_value)

#plotting the graph
fig = px.scatter(x=velocity_list, y=escaped_list)
fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(velocity_list), x1= max(velocity_list)
    )
])
fig.show()


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(escaped_list, (len(velocity_list), 1))
Y = np.reshape(velocity_list, (len(escaped_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='blue', linewidth=4)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

# do hit and trial by changing the value of X_test
plt.axvline(x=X_test[165], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
plt.show()
