import os
import tensorflow as tf
from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
from datetime import datetime, date, timedelta, time
from dateutil.relativedelta import relativedelta
import calendar
import numpy as np

# from .models import *
from .constants import *

def next_month(dt):
    dt = dt + timedelta(days=2)
    dt = dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
    return dt

def list_files(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

def dataframe_to_dataset(dataframe, y_var):
    dataframe = dataframe.copy()
    labels = dataframe.pop(y_var)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def previous_month(date):
    month, year = date.month - 1, date.year
    if month == 0:
        month = 12
        year = year - 1
    return month, year

def next_month_tuple(date):
    month, year = date.month + 1, date.year
    if month == 13:
        month = 1
        year = year + 1
    return month, year

def month_string(month_number):
    months = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }
    return months.get(month_number, 'Invalid Month')

def round_to_nearest_multiple(x, base=5):
    try:
        return base * round(x / base)
    except:
        return 0