from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pickle
import random

from starting_algorithm import gathering_data_set


def one_hot_encoder_applied():
    data_set, sales_price = gathering_data_set()

    year_one_hot_encoder = OneHotEncoder()
    make_one_hot_encoder = OneHotEncoder()
    body_type_one_hot_encoder = OneHotEncoder()
    model_one_hot_encoder = OneHotEncoder()

    mileage = np.array(data_set.mileage)
    engine_capacity = np.array(data_set.engine_capacity)
    year = np.array(data_set.year)

    encoded_year = year_one_hot_encoder.fit_transform(year.reshape(-1, 1)).toarray()
    make_en = np.array(data_set.make_en)
    encoded_make_en = make_one_hot_encoder.fit_transform(make_en.reshape(-1, 1)).toarray()

    body_en = np.array(data_set.body_type_en)
    encoded_body_en = body_type_one_hot_encoder.fit_transform(body_en.reshape(-1, 1)).toarray()
    model = np.array(data_set.model_en)

    encoded_model = model_one_hot_encoder.fit_transform(model.reshape(-1, 1)).toarray()
    mileage_engine_capacity = np.column_stack((mileage, engine_capacity.astype(np.float)))
    final_data_set = np.concatenate((encoded_year, encoded_make_en,
                                     encoded_body_en, encoded_model,
                                     mileage_engine_capacity), axis=1)

    dump_year_one_hot_encoder = open("dump_year_one_hot_encoder.pkl", "wb")
    pickle.dump(year_one_hot_encoder, dump_year_one_hot_encoder)
    dump_year_one_hot_encoder.close()

    dump_body_type_one_hot_encoder = open("dump_body_type_one_hot_encoder.pkl", "wb")
    pickle.dump(body_type_one_hot_encoder, dump_body_type_one_hot_encoder)
    dump_body_type_one_hot_encoder.close()

    dump_make_one_hot_encoder = open("dump_make_one_hot_encoder.pkl", "wb")
    pickle.dump(make_one_hot_encoder, dump_make_one_hot_encoder)
    dump_make_one_hot_encoder.close()

    dump_model_one_hot_encoder = open("dump_model_one_hot_encoder.pkl", "wb")
    pickle.dump(model_one_hot_encoder, dump_model_one_hot_encoder)
    dump_model_one_hot_encoder.close()

    return final_data_set, sales_price
