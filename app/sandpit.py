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


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

dir = "C:/Users/darre/PycharmProjects/projection/files/dummy/"
file = dir + "rate2.csv"
df = pd.read_csv(str(file))
df1 = pd.get_dummies(data=df, columns=['adviser_0'])
for col in df1: df1[col].replace({True: 1, False: 0}, inplace=True)
cols_to_scale = ['age_0', ]
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

x = df1.drop(['rate'], axis='columns')
y = df1['rate']

model = Sequential([
    Dense(20, input_shape=(x.shape[1],), activation='relu'),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile("adam", "binary_crossentropy", metrics=["MeanSquaredError"])
history = model.fit(x, y, epochs=150)
loss_array = history.history['loss']
loss = pd.DataFrame(loss_array, columns=['loss', ])


prediction = model.predict(x)
df['prediction'] = prediction
print(df)
# graph(df)
print(loss)
