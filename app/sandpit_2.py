import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def graph(df):
    fig, ax = plt.subplots()
    colours = ['green', 'blue', 'red']
    count = 0
    for adviser_0, group_df in df.groupby('adviser_0'):
        plt.plot(group_df['age_0'], group_df['rate'], label=f"Adviser {adviser_0}", color=colours[count], )
        plt.plot(group_df['age_0'], group_df['prediction'], label=f"Adviser {adviser_0}", linestyle='dotted', color=colours[count], )
        count += 1
    # Customize the plot
    plt.title("Rate vs. Age by Adviser")
    plt.xlabel("Age")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(False)
    plt.show()


# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns

dir = "C:/Users/darre/PycharmProjects/projection/files/rate/"
file = dir + "2023-06-30 to 2024-06-30.csv"
df = pd.read_csv(str(file))
df1 = pd.get_dummies(data=df, columns=['adviser_0'])
for col in df1: df1[col].replace({True: 1, False: 0}, inplace=True)
cols_to_scale = ['age_0', ]
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

x = df1.drop(['rate', 'cat_continuing', 'cat_exit'], axis='columns')
y = df1['rate']

model = Sequential([
    Dense(20, input_shape=(x.shape[1],), activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile("adam", "binary_crossentropy", metrics=["MeanSquaredError"])
model.fit(x, y, epochs=2000)

prediction = model.predict(x)
df['prediction'] = prediction
print(df)
graph(df)

