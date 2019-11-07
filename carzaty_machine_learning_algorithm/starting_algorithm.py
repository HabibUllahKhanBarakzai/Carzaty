import pandas as pd
import sqlalchemy
import pickle
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler, StandardScaler
import numpy as np
import math


def gathering_data_set():

    year_label_encoder = LabelEncoder()
    model_label_encoder = LabelEncoder()
    make_label_encoder = LabelEncoder()
    body_label_encoder = LabelEncoder()

    engine = sqlalchemy.create_engine("mysql+pymysql://habib:password@localhost:3306/carzaty")

    query = '''select make_en, model_en, year, (mileage) as mileage, (engine_capacity/3) as engine_capacity,
                body_type_en, sales_price
                from cars
                where condition_en = 'used'
                and make_en in (select make_en from cars group by make_en having count(*) > 450)
                and year in (2015, 2016, 2017, 2018, 2019)
                and mileage is not null
                and engine_capacity is not null
                and body_type_en is not null
                and sales_price is not null
                and model_en in (select model_en from cars group by model_en having count(*) > 900)'''

    data_set = pd.read_sql_query(query, engine)

    sales_price = np.array(data_set.sales_price).astype(np.float)
    data_set = data_set.drop(columns="sales_price")

    data_set.year = year_label_encoder.fit_transform(data_set.year)
    data_set.model_en = model_label_encoder.fit_transform(data_set.model_en)
    data_set.make_en = make_label_encoder.fit_transform(data_set.make_en)
    data_set.body_type_en = body_label_encoder.fit_transform(data_set.body_type_en)
    norm = np.array([data_set.mileage])
    normalizer_object = Normalizer().fit(norm)
    normalize_mileage = normalizer_object.transform(norm)

    norm_value = np.square(norm[0])
    norm_value = np.sum(norm_value)
    norm_value = math.sqrt(norm_value)
    final_array = norm[0]/norm_value
    data_set.mileage = final_array

    dump_year_label_encoder = open("year_label_encoder.pkl", 'wb')
    pickle.dump(year_label_encoder, dump_year_label_encoder)
    dump_year_label_encoder.close()

    dump_model_label_encoder = open("model_label_encoder.pkl", 'wb')
    pickle.dump(model_label_encoder, dump_model_label_encoder)
    dump_model_label_encoder.close()

    dump_make_label_encoder = open("make_label_encoder.pkl", 'wb')
    pickle.dump(make_label_encoder, dump_make_label_encoder)
    dump_make_label_encoder.close()

    dump_body_label_encoder = open("body_label_encoder.pkl", 'wb')
    pickle.dump(body_label_encoder, dump_body_label_encoder)
    dump_body_label_encoder.close()

    print("norm value is   ", norm_value)

    return data_set, sales_price


