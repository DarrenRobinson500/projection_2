import pandas as pd
from app.utilities import *

def lapse_run_logic(name, start, end):
    df1 = start.df()
    df2 = end.df()
    data_df = pd.merge(df1, df2, on=['number'], how='outer', indicator=True, suffixes=("_0", "_1"))
    lapse_df = data_df.loc[data_df['_merge'] == 'left_only']

    # filename = "files/" + name + ".csv"
    # lapse_df.to_csv(filename, index=False, header=True)
    # output_file = create_file(filename, "Lapse File")
    rate_df = calc_rates(data_df)

    lapse_analysis = LapseAnalysis(name=name)
    lapse_analysis.save()
    lapse_analysis.save_file(data_df, "start")
    lapse_analysis.save_file(data_df, "end")
    lapse_analysis.save_file(data_df, "data")
    lapse_analysis.save_file(rate_df, "rate")

    all_files = LapseAnalysis.objects.all()
    print("Post", all_files)

    return lapse_analysis

def calc_rates(df):
    grouped = df.groupby(["age_0", "adviser_0"]).count()
    survivors = grouped["age_1"]
    start = grouped["number"]
    rate = (1 - survivors / start)

    df = rate.to_frame()
    df.columns = ['rate',]
    df.to_csv("files/rate.csv", index=True, header=True)
    # rate_file = create_file("files/rate.csv", "Lapse Rate")
    df = pd.read_csv("files/rate.csv")

    return df