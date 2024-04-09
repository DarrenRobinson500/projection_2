# from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
# import tensorflow as tf

AGE_RANGE = range(20, 60)
ADVISER_RANGE = ["Adviser One", "Adviser Two", "Adviser Three"]

fn_data_0 = "files/data/30_Jun_2023.csv"
fn_data_1 = "files/data/31_Jul_2023.csv"
fn_data_2 = "files/data/31_Aug_2023.csv"
fn_data_c = "files/data_c.csv"
fn_lapse_rates = "files/rate.csv"
fn_lapse_table = 'files/lapse_table.csv'
fn_output = "files/output.csv"

def lapse_rate_calc(age, adviser):
    lapse_rate = age / 100
    if adviser == 3 and age > 45:
        lapse_rate *= 3
    lapse_rate = min(lapse_rate, 1)
    return lapse_rate

