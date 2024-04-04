# 1. Create the hypothetical data file and export to csv
from django.core.files import File

import pandas as pd
from app.constants import *
from app.models import *
from app.utilities import *

count_per_data_point = 10
count_per_data_point_new = 1

# Create a list for each file
def create_data_function():
    data = []
    fum = 10000
    dur = 5
    i = 1
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            for x in range(count_per_data_point):
                data.append((i, age, adviser, fum, dur))
                i += 1
    return data

def list_to_csv(list, filename):
    df = pd.DataFrame(data=list, index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
    print("list to csv:", filename, df.shape, len(list))
    df.to_csv(filename, index=False, header=True)
    directory = "C:/Users/darre/PycharmProjects/projection/files/"
    df1 = pd.read_csv(directory + "data_s.csv")
    print("list to csv:", filename, df.shape, len(list), df1.shape)


# Create records and save to csv
# def create_and_save_data():
#     data = []
#     fum = 10000
#     dur = 5
#     i = 1
#     for age in AGE_RANGE:
#         for adviser in ADVISER_RANGE:
#             for x in range(count_per_data_point):
#                 data.append((i, age, adviser, fum, dur))
#                 i += 1
#
#     name = "30_Jun_23"
#     df = pd.DataFrame(data=data, index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
#     file = create_file("data", name, df)
#     create_data_file(name=name, file=file)

def create_joint_file(data_0, data_1):
    # Create a combined csv (both start and end)
    df_s = pd.DataFrame(data=data_0, index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
    df_e = pd.DataFrame(data=data_1, index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
    df_joint = df_s.join(df_e.set_index('number'), on='number', how='outer', lsuffix='_0', rsuffix='_1')

    data_0_file = create_file(name="30_June_2023.csv", type="data", df=df_s)
    end_file = create_file(name="31_July_2023.csv", type="end", df=df_e)
    joint_file = create_file("July_23.csv", "joint", df=df_joint)
    start_date = datetime.strptime("30 June 2023", "%d %B %Y")
    end_date = datetime.strptime("31 July 2023", "%d %B %Y")

    Data(name="start", date=start_date, file=start_file).save()
    Data(name="end", date=end_date, file=end_file).save()


