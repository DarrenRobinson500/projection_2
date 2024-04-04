import math
from app.utilities import *
import numpy as np

def calc_lapse_rate(table, ages, adviser):
    result = []
    for age in ages:
        row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
        rate = row.iloc[0]['rate']
        result.append(rate)
    result = pd.Series(result)
    return result

def run_policy(lapse_analysis, data, time_period, detailed_output=False):

    # Data (for reading a series)
    age_s = data[f'age_{time_period}']
    print("Data:\n", data)
    print("Age_s:", age_s)
    fum_s = data[f'fum_s_{time_period}']
    adviser = data[f'adviser_{time_period}']

    if math.isnan(age_s):
        if time_period == 0: return pd.Series([0, 0, 0, 0])
        else:                return pd.Series([0, ])

    table = lapse_analysis.assumption.df()

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
    roll_forward = (pv_profit_0 - profit[0]) * discount_rate

    if time_period == 0:
        output = pd.Series([pv_profit_0,-profit[0],roll_forward,pv_profit_1_e])
    else:
        output = pd.Series([pv_profit_0,])

    if detailed_output:
        policies = pd.Series(policies)
        fum = pd.Series(fum)
        profit = pd.Series(profit)
        detail = pd.concat([policies, fum, profit], axis=1)
        # detail.rename({0: "policies", 1: "fum", 2: "profit"}, inplace=True)
        # print("Detail", detail)
        detail.to_csv(f"temp/projection_ind_{time_period}.csv")

    return output

def run_policy_aoc(info):
    info_0 = info[1:5]
    info_1 = info[5:9]

    general = General.objects.all().first()
    lapse_analysis = general.current_lapse_analysis

    # info_0.rename({"age_0": "age", "adviser_0": "adviser", "fum_s_0": "fum_s", "dur_s_0": "dur_s"}, inplace=True)
    output_0 = run_policy(lapse_analysis, info_0, time_period=0, detailed_output=general.detailed_output)

    # info_1.rename({"age_1": "age", "adviser_1": "adviser", "fum_s_1": "fum_s", "dur_s_1": "dur_s"}, inplace=True)
    output_1 = run_policy(lapse_analysis, info_1, time_period=1, detailed_output=general.detailed_output)

    output_0.rename({0: "Start", 1: "Less Profit", 2: "Rollforward", 3: "Expected"}, inplace=True)
    output_1.rename({0: "End"}, inplace=True)
    output_d = pd.Series([output_1["End"] - output_0["Expected"], f"http://127.0.0.1:8000/show/{info[0]}"])
    output_d.rename({0: "Delta", 1: "Link"}, inplace=True)
    output = pd.concat([output_0, output_1, output_d])

    return output

def get_category(info):
    age_s_0 = info['age_0']
    age_s_1 = info['age_1']
    if math.isnan(age_s_0): return "new"
    if math.isnan(age_s_1): return "exit"
    return "continuing"

def run_all_aoc(data):
    # Categorise each row
    category = data.apply(get_category, axis=1)
    data = pd.concat([data, category], axis=1)
    data.rename(columns={0: "cat",}, inplace=True)

    # Add calculated variables
    output = data.apply(run_policy_aoc, axis=1)
    data_cat_output = pd.concat([data, output], axis=1)

    # Save results
    general = General.objects.all().first()
    general.current_lapse_analysis.save_file(data_cat_output, "projection")

    # Turn off detailed results
    general.detailed_output=False
    general.save()

