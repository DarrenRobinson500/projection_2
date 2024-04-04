import pandas as pd
from keras import Input
from keras import Model as KerasModel
from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
import matplotlib.pyplot as plt
import mpld3

from app.utilities import *

def create_lapse_model(lapse_investigation):
    train_dataframe = lapse_investigation.rate.df()
    train_ds = dataframe_to_dataset(train_dataframe, "rate")

    # Create batches
    for x, y in train_ds.take(1):
        print("Input:", x)
        print("Target:", y)

    train_ds = train_ds.batch(32)

    # Encode the data
    age = Input(shape=(1,), name="age_0")
    adviser = Input(shape=(1,), name="adviser_0", dtype="int64")

    age_encoded = encode_numerical_feature(age, "age_0", train_ds)
    adviser_encoded = encode_categorical_feature(adviser, "adviser_0", train_ds, False)

    all_inputs = [age, adviser,]
    all_features = concatenate(
        [
            age_encoded,
            adviser_encoded,
        ]
    )

    # Define the  model
    x = Dense(32, activation="relu")(all_features)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = KerasModel(all_inputs, output)

    # Compile and run the model
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=lapse_investigation.epochs)

    # Plot the output vs expected
    # plot_data()

    # Save the model
    print(model.summary())
    filename = f"models/{lapse_investigation.name}.tf"
    lapse_investigation.model = filename
    lapse_investigation.save()
    if os.path.isfile(filename) is False:
        model.save(filename)


# train_dataframe = pd.read_csv("files/rate.csv")
# print(train_dataframe)

# train_ds = dataframe_to_dataset(train_dataframe)

def plot_data(lapse_investigation):
    if lapse_investigation.assumption is None: return
    model = lapse_investigation.get_model()
    fig, ax = plt.subplots()
    for adviser in ADVISER_RANGE:
        lapse_rates_model = create_lapse_rate_model(lapse_investigation, adviser)
        lapse_rates_actual = create_lapse_rate_actual(adviser)
        label_m = f"Adviser {adviser} (modelled)"
        label_a = f"Adviser {adviser} (actual)"
        ax.plot(AGE_RANGE, lapse_rates_model, label=label_m)
        ax.plot(AGE_RANGE, lapse_rates_actual, label=label_a)
    # fig.title("Lapse rates")
    # fig.xlabel("Age")
    # fig.ylabel("Lapse rate")
    # fig.legend()
    # fig.show()
    html = mpld3.fig_to_html(fig)
    return html

def create_lapse_rate_model_calc(model, adviser):
    print("\nGraph")
    lapse_rate_model = []
    for age in AGE_RANGE:
        sample = {"age_0": age, "adviser_0": adviser, }
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        # print()
        prediction = model.predict(input_dict)[0][0]
        print("Create lapse rate model", input_dict, prediction)
        lapse_rate_model.append(prediction)
    print(lapse_rate_model)
    return lapse_rate_model

def create_lapse_rate_actual(adviser):
    lapse_rate_actual = []
    for age in AGE_RANGE:
        lapse_rate_actual.append(lapse_rate_calc(age, adviser))
    return lapse_rate_actual

def create_lapse_rate_model(lapse_analysis, adviser):
    df = lapse_analysis.assumption.df()
    result = []
    for age in AGE_RANGE:
        filtered_df = df[(df['age_0'] == age) & (df['adviser_0'] == adviser)]
        result.append(filtered_df.iloc[0]['rate'])
    return result


def create_lapse_assumption_table(lapse_analysis):
    print("\nAssumption Table")
    model = lapse_analysis.get_model()
    if model is None: return

    AGE_RANGE = range(20,66)
    ADVISER_RANGE = [1, 2, 3]

    table = []
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            sample = {"age_0": age, "adviser_0": adviser, }
            input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
            prediction = round(model.predict(input_dict)[0][0], 3)
            row = (age, adviser, prediction)
            print("Create lapse rate assumption", input_dict, prediction)
            table.append(row)
    dataframe = pd.DataFrame(data=table, index=None, columns=['age_0', 'adviser_0', 'rate'])
    lapse_analysis.save_file(dataframe, "assumption")

