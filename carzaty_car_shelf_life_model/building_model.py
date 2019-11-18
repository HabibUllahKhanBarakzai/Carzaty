import pickle
import numpy as np

with open("mileage_scaled.pkl", "rb") as input_file:
    mileage_scaled = pickle.load(input_file)

t = [32200]
data = np.array([t]).reshape(-1, 1)
print(mileage_scaled.transform(data))
