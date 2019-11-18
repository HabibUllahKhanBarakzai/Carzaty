from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import sqlalchemy
import pickle
from collections import OrderedDict


engine = sqlalchemy.create_engine("mysql+pymysql://habib:password@localhost:3306/carzaty")

query = """select year, make_en, model_en,
          body_type_en, engine_capacity, engine_cylinders, mileage, life 
          from MY_TABLE
            where make_en in
          (select make_en from MY_TABLE group by make_en having Count(*) > 50)
          and model_en in (select model_en from MY_TABLE group by model_en having COUNT(*) > 40)
            and year is not null
            and make_en is not null
            and model_en is not null
            and body_type_en is not null
            and engine_capacity is not null
            and engine_cylinders is not null
            and life is not null
"""

data_set = pd.read_sql_query(query, engine)
data_set = data_set.dropna().reset_index(drop=True)

columns_to_be_encoded = ["make_en", "model_en", "body_type_en", "engine_cylinders"]
columns_to_be_scaled = ["mileage", "engine_capacity"]
encoding_objects = OrderedDict()
encoded_columns = OrderedDict()
final_encoded_dictionary = OrderedDict()


for value in columns_to_be_encoded:
    encoding_objects["{}_label_encoded".format(value)] = LabelEncoder()
    encoding_objects["{}_one_hot_encoding".format(value)] = OneHotEncoder()

    data_to_be_transformed = np.array([data_set[value]]).reshape(-1, 1)

    final_encoded_dictionary[value] = encoding_objects["{}_one_hot_encoding".format(value)].\
        fit_transform(data_to_be_transformed).toarray()

    # with open("{}_one_hot_encoding.pkl".format(value), "wb") as one_hot_output:
    #     pickle.dump(encoding_objects["{}_one_hot_encoding".format(value)], one_hot_output, pickle.HIGHEST_PROTOCOL)

for value in columns_to_be_scaled:
    encoding_objects["{}_scaled".format(value)] = MinMaxScaler()
    final_encoded_dictionary[value] = encoding_objects["{}_scaled".format(value)].\
        fit_transform(np.array([data_set[value]]).reshape(-1, 1))

    # with open("{}_scaled.pkl".format(value), "wb") as output:
    #     pickle.dump(encoding_objects["{}_scaled".format(value)], output, pickle.HIGHEST_PROTOCOL)

for value in final_encoded_dictionary:
    if value in columns_to_be_encoded:
        final_encoded_dictionary[value] = final_encoded_dictionary[value][:,
                                          0:len(final_encoded_dictionary[value][0]) - 1]


training_set = np.array(final_encoded_dictionary["make_en"])


for value in final_encoded_dictionary:
    if value != "make_en":
        training_set = np.concatenate((training_set, final_encoded_dictionary[value]),
                                       axis=1)

x_train, x_test, y_train, y_test = train_test_split(training_set,
                                                    np.array(data_set.life), test_size=0.05)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adamax, Adagrad, Adam

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=x_train[0].shape))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="relu"))

optimizer = Adagrad(lr=0.008, decay=0.000001)

model.compile(metrics=["mae"], loss="logcosh",
              optimizer=optimizer)

model.fit(x_train, y_train, epochs=50, validation_split=0.03)

prediction = model.predict(x_test)
print(r2_score(prediction, y_test))

for i  in range(100):
    print(prediction[i], "            ", y_test[i])


