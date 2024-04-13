import pandas as pd
import numpy as np
import math

proj_period = 10


def projection(data, time_period):
    age_s = data[f'age_{time_period}']
    fum_s = data[f'fum_s_{time_period}']
    adviser = data[f'adviser_{time_period}']

    # Time
    time = np.arange(proj_period)
    age = age_s + time
    time = pd.Series(time)
    age = pd.Series(age)
    discount_rate = 0.10
    inv_earnings = 0.07
    disc_factor = (1 / (1 + discount_rate)) ** time


    df = pd.concat([time, age], axis=1)
    df.rename({0:"time", 1: "age"}, inplace=True, axis=1)
    df['adviser_0'] = adviser
    df['age_0'] = (df['age'] // 5) * 5
    df = pd.merge(df, table[['age_0', 'adviser_0', 'rate']], on=['age_0', 'adviser_0'], how='left')
    df.drop('age', inplace=True, axis=1)
    df['count'] = 1
    df['count'] = df['count'].astype('float64')
    for i in range(1, len(df)): df.loc[i, 'count'] = df.loc[i - 1, 'count'] * (1 - df.loc[i, 'rate'])
    df['fum'] = 1
    df['fum'] = df['fum'].astype('float64')
    for i in range(1, len(df)): df.loc[i, 'fum'] = df.loc[i - 1, 'fum'] * (1 + inv_earnings)
    df['disc'] = 1
    df['disc'] = df['disc'].astype('float64')
    for i in range(1, len(df)): df.loc[i, 'disc'] = df.loc[i - 1, 'disc'] / (1 + discount_rate)
    df['value'] = df['count'] * df['fum'] * df['disc']

    return df['value'].sum()




dir = "C:/Users/darre/PycharmProjects/projection/files/"
file = dir + "dummy/single_policy.csv"
df = pd.read_csv(str(file))
file = dir + "assumption/FY24.csv"
table = pd.read_csv(str(file))

# print(df)
df1 = df.apply(projection, time_period=0, axis=1)
df1 = pd.concat([df, df1], axis=1)
df1.rename({0: "value"}, inplace=True, axis=1)

# print(table)
print(df1)

