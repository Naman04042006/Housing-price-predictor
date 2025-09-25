import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    "area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
    "price": [50, 60, 75, 90, 110, 120, 135, 150, 165, 180]  # in lakhs
}

df = pd.DataFrame(data)


X = df[['area']]   
y = df[['price']]  


model = LinearRegression()
model.fit(X, y)


new_area = [[2300]] 
predicted_price = model.predict(new_area)

print(f"Predicted price for {new_area[0][0]} sq ft: {predicted_price[0][0]:.2f} lakhs")


plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (lakhs)")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.show()
