import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

proj_period = 10


def plot(df, column):
    fig, ax = plt.subplots()
    colours = ['green', 'blue', 'red']
    count = 0
    for adviser_0, group_df in df.groupby('adviser_0'):
        plt.plot(group_df['age_0'], group_df[column], label=f"Adviser {adviser_0}", color=colours[count])
        count += 1

    # Customize the plot
    plt.title("Value vs. Age by Adviser")
    plt.xlabel("Age")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(False)
    plt.show()


def apply_value(data, table):
    age_s = data['age_0']
    adviser = data['adviser_0']

    # Time
    time = np.arange(proj_period)
    age = age_s + time
    time = pd.Series(time)
    age = pd.Series(age)
    discount_rate = 0.10
    inv_earnings = 0.07

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

def apply_value_to_data(df, table):
    df = pd.merge(df, table[['age_0', 'adviser_0', 'rate', 'value']], on=['age_0', 'adviser_0'], how='left')
    # df.drop('age', inplace=True, axis=1)
    # df['count'] = 1
    # df['count'] = df['count'].astype('float64')
    # for i in range(1, len(df)): df.loc[i, 'count'] = df.loc[i - 1, 'count'] * (1 - df.loc[i, 'rate'])
    # df['fum'] = 1
    # df['fum'] = df['fum'].astype('float64')
    # for i in range(1, len(df)): df.loc[i, 'fum'] = df.loc[i - 1, 'fum'] * (1 + inv_earnings)
    # df['disc'] = 1
    # df['disc'] = df['disc'].astype('float64')
    # for i in range(1, len(df)): df.loc[i, 'disc'] = df.loc[i - 1, 'disc'] / (1 + discount_rate)
    # df['value'] = df['count'] * df['fum'] * df['disc']

    return df



dir = "C:/Users/darre/PycharmProjects/projection/files/"
file = dir + "dummy/single_policy.csv"
df = pd.read_csv(str(file))
file = dir + "assumption/FY24.csv"
table = pd.read_csv(str(file))

if 'value' not in table.columns:
    value = table.apply(apply_value, table=table, axis=1)
    table = pd.concat([table, value], axis=1)
    table.rename({0: "value"}, inplace=True, axis=1)

# print(table)
print(table)

df = pd.merge(df, table[['age_0', 'adviser_0', 'rate', 'value']], on=['age_0', 'adviser_0'], how='left')
print(df)

# plot(table1, "value")

