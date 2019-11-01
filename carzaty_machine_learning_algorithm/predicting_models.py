import keras
import pickle
import numpy as np

model = keras.models.load_model("car_prediction_model_1.h5")

with open("body_label_encoder.pkl", "rb") as input_file:
    body_label_encoder = pickle.load(input_file)

with open("make_label_encoder.pkl", "rb") as input_file:
    make_label_encoder = pickle.load(input_file)

with open("model_label_encoder.pkl", "rb") as input_file:
    model_label_encoder = pickle.load(input_file)

with open("year_label_encoder.pkl", "rb") as input_file:
    year_label_encoder = pickle.load(input_file)

with open("dump_year_one_hot_encoder.pkl", "rb") as input_file:
    year_one_hot_encoder = pickle.load(input_file)

with open("dump_model_one_hot_encoder.pkl", "rb") as input_file:
    model_one_hot_encoder = pickle.load(input_file)

with open("dump_make_one_hot_encoder.pkl", "rb") as input_file:
    make_one_hot_encoder = pickle.load(input_file)

with open("dump_body_type_one_hot_encoder.pkl", "rb") as input_file:
    body_type_one_hot_encoder = pickle.load(input_file)

mileage_value = 5000/380
engine_value = 1.8/3
prediction_value = ["Mercedes-Benz", "C Class", 2019, mileage_value, engine_value, "Convertible"]

year = year_label_encoder.transform([prediction_value[2]])
year = year_one_hot_encoder.transform([year]).toarray()
make_en = make_label_encoder.transform([prediction_value[0]])
make_en = make_one_hot_encoder.transform([make_en]).toarray()

model_en = model_label_encoder.transform([prediction_value[1]])
model_en = model_one_hot_encoder.transform([model_en]).toarray()


body = body_label_encoder.transform([prediction_value[5]])
body = body_type_one_hot_encoder.transform([body]).toarray()

mileage = np.array([prediction_value[3]])
engine = np.array([prediction_value[4]])
mileage_and_engine_capacity = np.column_stack((mileage, engine))
final_data = np.concatenate((year, make_en, body, model_en, mileage_and_engine_capacity), axis=1)

prediction = model.predict(final_data)
print(prediction)