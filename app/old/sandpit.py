import math
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import Input

# from utilities import *

filename = "/files/lapse_1/5_assumption.csv"
table = pd.read_csv(str(filename))
# print("Table:", table)

def calc_lapse_rate(table, ages, adviser):
    result = []
    for age in ages:
        row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
        rate = row.iloc[0]['rate']
        result.append(rate)
    result = pd.Series(result)
    return result

def run_policy(age_s, fum_s, adviser, time_period):

    if math.isnan(age_s):
        if time_period == 0:
            output = pd.Series([0, 0, 0, 0])
        else:
            output = pd.Series([0, ])
        return output

    # Time
    time = np.arange(65-age_s)
    age = age_s + time

    # Economic Assumptions
    discount_rate = 0.10
    inv_earnings = 0.07

    disc_factor = (1 / (1 + discount_rate)) ** time

    # Policy Assumptions
    lapse_rate = calc_lapse_rate(table, age, adviser)

    # Product Information
    fee_rate = 0.02
    expense_rate = 0.01

    # Calcs
    policies = (1 - lapse_rate) ** time
    fum_pp_factor = (1 + inv_earnings) ** time
    fum_pp = fum_s * fum_pp_factor
    fum = fum_pp * policies
    fees = fum * fee_rate
    expense = fum * expense_rate
    profit = fees - expense
    profit_disc = profit * disc_factor
    fees_disc = fees * disc_factor
    pv_fees = round(sum(fees_disc),2)
    pv_profit_0 = round(sum(profit_disc),2)
    pv_profit_1_e = round(sum(profit_disc[1:]),2) * (1 + discount_rate)
    # aop_pv_profit_0 = pv_profit
    # aop_pv_profit_0 = pv_profit
    # print(pv_profit_0, profit[0], pv_profit_1)
    roll_forward = (pv_profit_0 - profit[0]) * discount_rate

    if time_period == 0:
        output = pd.Series([pv_profit_0,-profit[0],roll_forward,pv_profit_1_e])
    else:
        output = pd.Series([pv_profit_0,])

    # aop = pd.DataFrame.from_dict(data, orient='index', columns=['Values'])
    # print(aop)

    # result = pd.Series([pv_fees, pv_profit])
    return output


def run_policy_aoc(data):
    # print("Run policy aoc - cat:", data["cat"])
    # if data["cat"] == "exit": return
    data_0 = data[1:5]
    data_1 = data[5:9]

    output_0 = run_policy(age_s=data_0["age_0"], fum_s=data_0["fum_s_0"], adviser=data_0["adviser_0"], time_period=0)
    output_1 = run_policy(age_s=data_1["age_1"], fum_s=data_1["fum_s_1"], adviser=data_1["adviser_1"], time_period=1)
    output_0.rename({0: "Start", 1: "Less Profit", 2: "Rollforward", 3: "Expected"}, inplace=True)
    output_1.rename({0: "End"}, inplace=True)
    output_d = pd.Series([output_1["End"] - output_0["Expected"],])
    output_d.rename({0: "Delta"}, inplace=True)
    output = pd.concat([output_0, output_1, output_d])

    # delta = output["End"] - output["Expected"]

    # output_dict = output_0_dict | output_1_dict
    # output = pd.DataFrame.from_dict(output_dict, orient='index', columns=['Values']).transpose()
    # print("\nOutput")
    # print(output[0])

    # output = [output_dict['Start'], output_dict["End"]]
    # print(output)

    return output

def get_category(info):
    age_s_0 = info['age_0']
    age_s_1 = info['age_1']
    if math.isnan(age_s_0): return "new"
    if math.isnan(age_s_1): return "exit"
    return "continuing"

def run_all_aoc(data):
    # Get the data
    # data = pd.read_csv("files/data.csv")

    # Categorise each row
    category = data.apply(get_category, axis=1)
    data_cat = pd.concat([data, category], axis=1)
    data_cat.rename(columns={0: "cat",}, inplace=True)


    # Add pv_profit and pv_fees
    output = data_cat.apply(run_policy_aoc, axis=1)
    data_cat_output = pd.concat([data_cat, output], axis=1)
    # print("Data cat output")
    print(data_cat_output.to_string())
    # data_cat_output['profit_d'] = data_cat_output['End'] - data_cat_output['Expected']
    # print(data_cat_output)

    # Group the data
    grouping = ["cat", "adviser_0", ]
    grouping = ["cat", ]
    grouped = data_cat_output.groupby(grouping)
    pd.options.display.float_format = '{:,.0f}'.format
    # value_0 = grouped['profit_0'].sum()
    # value_1 = grouped['profit_1'].sum()
    # count_b = grouped['profit_0', 'profit_1'].count()

    # print("Value 0", value_0)
    # print("Value 1", value_1)

    # value_b = grouped['profit_0', 'profit_1', 'profit_d'].sum()
    # value_b.append(value_b.sum(numeric_only=True), ignore_index=True)
    # value_b.loc['total'] = value_b.sum()

    # Print the summary
    # print()
    # print("SUMMARY")
    # print(count_b)
    # print()
    # print(value_b)

    # print(f'Start value: {sum(value_0):,.0f}')
    # print(f'End value: {sum(value_1):,.0f}')
    # print(f'Change in value: {sum(value_1 - value_0):,.0f}')



filename = "/files/data.csv"
data = pd.read_csv(filename)
# print(data)
run_all_aoc(data)

# result = run_policy_aoc()
# print(result)