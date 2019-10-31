from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax
from keras.layers.advanced_activations import LeakyReLU

from applying_one_hot_encoding import one_hot_encoder_applied

X, y = one_hot_encoder_applied()

model = Sequential()
model.add(Dense(500, input_shape=X.shape[1:], activation="relu"))
model.add(Dense(220,  activation="relu"))
model.add(Dense(220,  activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1,))

optimizer = Adamax(learning_rate=0.01)

model.compile(optimizer=optimizer,
              metrics=["mae"],
              loss="mean_squared_logarithmic_error")

model.fit(X, y,
          epochs=5000)

model.save("car_prediction_model_1.h5")
