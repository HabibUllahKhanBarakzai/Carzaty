from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_squared_error
import keras

from applying_one_hot_encoding import one_hot_encoder_applied
X, y = one_hot_encoder_applied()

leaky_relu = keras.layers.LeakyReLU(alpha=0.3)


model = Sequential()
model.add(Dense(100, input_shape=X.shape[1:], activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(22,  activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(22,  activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(10, activation="relu"))
model.add(Dense(1,))

optimizer = Adamax(learning_rate=0.1)

model.compile(optimizer=optimizer,
              metrics=["mae"],
              loss="logcosh")

model.fit(X, y,
          epochs=50, validation_split=0.01)

model.save("car_prediction_model_1.h5")
