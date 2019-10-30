from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.advanced_activations import LeakyReLU

from applying_one_hot_encoding import one_hot_encoder_applied

X, y = one_hot_encoder_applied()

model = Sequential()
model.add(Dense(1000, input_shape=X.shape[1:], activation="relu"))
model.add(Dropout(0.15))
model.add(Dense(500,  activation="relu"))
model.add(Dropout(0.12))
model.add(Dense(500,  activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(220,  activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(200,  activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.01, decay=0.01)

model.compile(optimizer=optimizer,
              metrics=["mape"],
              loss="mean_squared_logarithmic_error")

model.fit(X, y,
          epochs=100, validation_split=0.15)

model.save("car_prediction_model_1.h5")
