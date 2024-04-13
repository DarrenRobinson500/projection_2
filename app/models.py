from keras.models import load_model
from .utilities import *
from keras import Input
from keras import Model as KerasModel
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import mpld3
import math, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import pandas as pd
from django.db.models import *
from django.core.files import File

def format_2dp(var): return f'{var:,.2f}'
def format_age(var): return f'{var:.0f}'
def format_int(var): return f'{var:.0f}'
def format_adviser(var): return f'{var:.0f}'
def format_dollars(var): return f'{var:,.0f}'
def format_percent(var): return f"{var:.1%}"

formatters = {
    'Time': format_int,
    'Policies': format_2dp,
    'FUM': format_dollars, 'fum_s': format_dollars, 'fum_s_0': format_dollars, 'fum_s_1': format_dollars,
    'Profit': format_2dp,
    'value': format_2dp,
    'dur_s_0': format_int, 'dur_s_1': format_int,
    'rate': format_percent,
    'age_0': format_int, 'age_1': format_int,
    # 'adviser_0': format_int,'adviser_1': format_int,
    'Start': format_dollars, 'Less Profit': format_dollars, 'Rollforward': format_dollars,
    'Expected': format_dollars, 'End': format_dollars, 'Delta': format_dollars,
}

def create_file(type, name, df, index=False):
    file_name = f"files/{type}/{name}.csv"

    directory = f"files/{type}"
    if not os.path.exists(directory): os.makedirs(directory)

    existing_files = FileModel.objects.filter(name=file_name)
    existing_files.delete()

    df.to_csv(file_name, index=index, header=True)
    File = FileModel(name=file_name, type=type)
    File.save()
    return File

class FileModel(Model):
    TYPE_CHOICES = [
        ("start", "Start File"),
        ("end", "End File"),
        ("data", "Data File"),
        ("joint", "Joint File"),
        ("lapse", "Lapse File"),
        ("rate", "Rate File"),
        ("assumption", "Assumption File"),
        ("projection", "Projection File"),
    ]
    name = "file"
    name_singular = "File"
    name_plural = "Files"
    description = "Data, Joins, Periods and DNNs each create files. Those files are listed below. To view a file, click on the relevant button."
    name = CharField(max_length=512)
    time_stamp = DateTimeField(auto_now_add=True, null=True,blank=True)
    last_update = DateTimeField(null=True,blank=True)
    document = FileField(upload_to="files/", blank=True, null=True)
    type = CharField(max_length=100, blank=True, null=True, choices=TYPE_CHOICES)

    def __str__(self):
        if self.name: return self.name
        else: "No name given"

        return self.name

    def col_types(self):
        df = self.df()
        if df is None: return
        # result = []
        # for col in df:
        #     result.append((col, col.dtype))
        print("Col types:", type(df.dtypes))
        return df.dtypes

    def owner(self):
        if self.type == "data": model = Data
        elif self.type == "join": model = Join
        elif self.type == "period": model = Period
        elif self.type.lower() == "dnn": model = DNN
        elif self.type == "projection": model = Projection
        else: return None
        return model.objects.filter(file=self).first()

    def get_max(self, header_str):
        df = self.df()
        value = df[header_str].max()
        return value

    def df(self):
        try:
            return pd.read_csv(str(self.name))
        except:
            return None

    def df_html(self):
        df = self.df()
        if df is None: return None
        result = df.to_html(
            classes=['table', 'table-striped', 'table-center'],
            index=False,
            justify='center',
            formatters=formatters,
            render_links=True,
        )
        return result

    def df_total(self):
        if self.type != "data": return
        df = self.df()

        grouping = "adviser"
        df_group = df.groupby([grouping]).sum()
        df_group = df_group[["fum_s", ]]
        df_total = df_group.sum().to_frame().transpose()
        df_total.set_index(pd.Index(["Total", ]), inplace=True)
        df = pd.concat([df_group, df_total], axis=0)

        html = f"<h3>Total</h3>" + df.to_html(classes=['table', 'table-striped', 'table-center'], index=True, justify='center', formatters=formatters)
        return html

    def df_extra_html(self):
        if self.type != "projection": return
        df = self.df()
        html = ""

        # for heading, grouping in [("Advisers", 'adviser_0'), ]:

        for heading, grouping in [("Advisers", 'adviser_0'), ("Category", 'cat')]:
            df_group = df.groupby([grouping]).sum()
            df_group = df_group[["Start", "Less Profit", "Rollforward", "Expected", "End", "Delta"]]
            df_total = df_group.sum().to_frame().transpose()
            df_total.set_index(pd.Index(["Total",]), inplace=True)
            df_total = pd.concat([df_group, df_total], axis=0)

            html += f"<h3>{heading}</h3>" + df_total.to_html(classes=['table', 'table-striped', 'table-center'], index=True, justify='center', formatters=formatters)
        # html += f"<h3>All Records</h3>"
        # html.replace(">http://127.0.0.1:8000/show/", "Show ")

        return html

    def df_top(self):
        try:
            df = pd.read_csv(str(self.name), nrows=4)
            return df
        except:
            return

    def df_top_html(self):
        try:
            return self.df_top().to_html(classes=['table', 'table-striped', 'table-center'], index=False, justify='center', formatters=formatters, render_links=True,)
        except:
            return

    def delete(self, *args, **kwargs):
        try:
            print(f"Deleted: {self.name}")
            os.remove(self.name)
        except FileNotFoundError:
            print(f"File not found: {self.name}")
        super().delete(*args, **kwargs)

def create_first_data():
    data = []
    fum = 10000
    dur = 5
    i = 1
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            for x in range(10):
                data.append((i, age, adviser, fum * age, dur))
                i += 1

    name = "30_Jun_23"
    df = pd.DataFrame(data=data, index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
    file = create_file("data", name, df)
    create_data_file(name=name, file=file)

def create_data_file(name, file):
    date = datetime.strptime(name, "%d_%b_%y")
    existing_files = Data.objects.filter(name=name, date=date)
    existing_files.delete()
    index = file.get_max('number')
    new_data = Data(name=name, date=date, file=file, index=index)
    new_data.save()

    # Prior Join
    month, year = previous_month(date)
    prior_data = Data.objects.filter(Q(date__year=year, date__month=month)).first()
    if prior_data:
        join = Join(start=prior_data, end=new_data, name=new_data.month_string())
        join.save()
        join.create_file()

    # Next Joint
    month, year = next_month_tuple(date)
    next_data = Data.objects.filter(Q(date__year=year, date__month=month)).first()
    if next_data:
        join = Join(start=new_data, end=next_data, name=next_data.month_string())
        join.save()
        join.create_file()

class Data(Model):
    name = "data"
    name_singular = "Data"
    name_plural = "Data"
    description = "The Data file contains a snapshot of the portfolio at the end of the relevant month."

    name = CharField(max_length=512, null=True)
    date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="data_file", on_delete=SET_NULL, null=True)
    df = None
    index = IntegerField(null=True, blank=True)
    value_factor = FloatField(null=True, blank=True)

    def __str__(self):
        if self.name: return self.name
        else: "No name given"

    def apply_value(self, table):
        df = self.file.df()
        for column in ['rate', 'value', 'rate_x', 'value_x', 'rate_y', 'value_y', ]:
            if column in df.columns: df.drop(columns=[column, ], axis=1, inplace=True)
        df['age_0'] = (df['age'] // 5) * 5
        df['adviser_0'] = df['adviser']
        df = pd.merge(df, table[['age_0', 'adviser_0', 'rate', 'value']], on=['age_0', 'adviser_0'], how='left')
        df.drop(['age_0', 'adviser_0'], axis=1)

        self.value_factor = round((df['fum_s'] * df['value']).sum() / df['fum_s'].sum(), 2)
        self.save()

        file_name = f"files/data/{self.name}.csv"
        df.to_csv(file_name, index=False, header=True)

    def join(self):
        return Join.objects.filter(end=self).first()

    def month_string(self):
        return f"{month_string(self.date.month)} {self.date.year}"

    def create_next(self):
        input_df = self.file.df()
        rates = input_df.apply(self.add_lapse_rates, axis=1)
        rates.name = "Action"
        df = pd.concat([input_df, rates], axis=1)
        df = df[df['Action'] == "Keep"]
        df.drop(columns=['Action'], inplace=True)
        df = self.add_new_business(df)
        df = self.add_investment_return(df)
        dt = next_month(self.date)
        name = dt.strftime("%d_%b_%y")
        file = create_file("data", name, df)
        create_data_file(name=name, file=file)

    def add_lapse_rates(self, data):
        age = data[f'age']
        adviser = data[f'adviser']

        setting = Setting.objects.all().first()
        projection = setting.projection
        try:
            table = projection.dnn.assumption_file.df()
        except:
            table = pd.read_csv(str("files/dummy/rate.csv"))
            # print("Used dummy data")
        # print("Add lapse rates table:\n", table)
        row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
        annual_rate = row.iloc[0]['rate']
        rate = 1 - (1 - annual_rate) ** (1/12)
        # print("Rate:", annual_rate, rate)
        random_number = random.random()
        if random_number < rate: return "Delete"
        return "Keep"

    def add_new_business(self, df):
        if not self.index:
            index = 1
        else:
            index = self.index + 1
        fum, duration = 100000, 0
        for age in AGE_RANGE:
            for adviser in ADVISER_RANGE:
                    index += 1
                    # new_row = {'number': index, 'age': age, 'adviser': adviser, 'fum': fum}
                    df.loc[-1] = [index, age, adviser, fum, duration]
                    # df = df.append(new_row, ignore_index=True)
        self.index = index
        self.save()
        return df

    def add_investment_return(self, df):
        inv_return = random.randint(-3, 3) / 100
        df['fum_s'] = df['fum_s'] * ( 1 + inv_return)
        return df

    def files(self):
        return [self.file, ]

    def delete(self, *args, **kwargs):
        for file in self.files():
            if file:
                file.delete()
        super().delete(*args, **kwargs)

class Join(Model):
    name = "join"
    name_singular = "Join"
    name_plural = "Joins"
    description = "The Join file combines the Data file from the end of the previous month and the end of the current month. The change during the month for each record can be seen in the Join file. Joins are made automatically when Data files are created."

    name = CharField(max_length=10, null=True)
    start = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="start")
    end = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="end")
    end_date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="joint_file", on_delete=SET_NULL, null=True)
    def __str__(self):
        if self.end:
            return self.end.month_string()
        return "No end file"

    def df(self):
        if self.start is None or self.start.file.df() is None:
            print("Problem with start file")
            return None
        if self.end is None or self.end.file.df() is None:
            print("Problem with end file")
            return None
        df_s = pd.DataFrame(data=self.start.file.df(), index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
        df_e = pd.DataFrame(data=self.end.file.df(), index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
        df = df_s.join(df_e.set_index('number'), on='number', how='outer', lsuffix='_0', rsuffix='_1')
        df = df[~df.index.duplicated(keep='first')]
        df['adviser_0'] = df['adviser_0'].combine_first(df['adviser_1'])

        # Categorise each row
        category = df.apply(get_category, axis=1)
        df = pd.concat([df, category], axis=1)
        df.rename(columns={0: "cat", }, inplace=True)

        return df

    def df_rec(self):
        df = self.file.df()

        df_group = df.groupby(['cat']).sum()
        df_group = df_group[["fum_s_0", "fum_s_1", ]]
        df_total = df_group.sum().to_frame().transpose()
        df_total.set_index(pd.Index(["Total",]), inplace=True)
        df = pd.concat([df_group, df_total], axis=0)

        html = df.to_html(classes=['table', 'table-striped', 'table-center'], index=True, justify='center', formatters=formatters)

        return html

    def create_file(self):
        self.file = create_file("join", f"{self}", self.df())
        self.end_date = self.end.date
        self.save()

    def files(self):
        return [self.file, ]

    def delete(self, *args, **kwargs):
        for file in self.files():
            if file:
                file.delete()
        super().delete(*args, **kwargs)

class Period(Model):
    name = "period"
    name_singular = "Period"
    name_plural = "Periods"
    description = "A Period contains all Joins with dates between the Period's start and end date."

    name = CharField(max_length=10, null=True)
    start_date = DateField(null=True, blank=True)
    end_date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="period_file", on_delete=SET_NULL, null=True)
    rate_file = ForeignKey(FileModel, related_name="rate_file", on_delete=SET_NULL, null=True)
    # encoded_rate_file = ForeignKey(FileModel, related_name="encoded_rate_file", on_delete=SET_NULL, null=True)
    def __str__(self): return f"{self.start_date} to {self.end_date}"

    def joins(self):
        return Join.objects.filter(end_date__range=[self.start_date, self.end_date])

    def df(self):
        joins = self.joins()
        # print("Len of joins:", len(joins))
        if len(joins) == 0: return None
        for count, join in enumerate(joins):
            if count == 0: df = join.df()
            else:
                df_new = join.df()
                if df_new is not None:
                    df = pd.concat([df, df_new], ignore_index=True)
        return df

    def create_rate_file(self):
        df = self.df()

        df['age_0'] = df['age_0'].apply(lambda x: round_to_nearest_multiple(x, base=5))
        df['age_1'] = df['age_1'].apply(lambda x: round_to_nearest_multiple(x, base=5))
        df = pd.get_dummies(data=df, columns=['cat'])

        rate = df.groupby(["age_0", "adviser_0"]).sum()
        rate['rate'] = 1 - (1 - rate['cat_exit'] / rate['cat_continuing']) ** 12
        required_columns = ['cat_continuing', 'cat_exit', 'rate']
        rate = rate[required_columns]
        rate = rate[rate['rate'].notna()]

        rate.to_csv("files/rate.csv", index=True, header=True)
        rate = pd.read_csv("files/rate.csv")
        self.rate_file = create_file("rate", f"{self}", rate)
        self.save()

    def create_files(self):
        # Add all the joins together
        self.file = create_file("period", f"{self}", self.df())
        self.save()
        # Create the rates
        self.create_rate_file()
        # Encode the data
        # self.create_encoded_rate_file()

    def files(self):
        return [self.file, self.rate_file]

    def delete(self, *args, **kwargs):
        for file in self.files():
            if file:
                file.delete()
        super().delete(*args, **kwargs)

class DNN(Model):
    name = "dnn"
    name_singular = "DNN"
    name_plural = "DNNs"
    description = "A Deep Neural Network (DNN) is defined by the number and size of its layers. Larger DNNs are more able to model complexity and more prone to 'over-fitting' the data. Each DNN is connected to a Period. The Joins in the Period are used to train the model."

    name = CharField(max_length=30, null=True)
    period = ForeignKey(Period, related_name="period", on_delete=SET_NULL, null=True)
    layers = CharField(max_length=512, null=True, default='20, 15, 10, 10, 5')
    model = CharField(max_length=512, null=True)
    epochs = IntegerField(default=150)
    assumption_file = ForeignKey(FileModel, related_name="assumption", on_delete=SET_NULL, null=True)
    loss_file = ForeignKey(FileModel, related_name="loss", on_delete=SET_NULL, null=True)
    proj_period = IntegerField(blank=True, null=True, default=5)
    mse = FloatField(null=True, blank=True)
    def __str__(self): return self.name

    def files(self):
        return [self.assumption_file, self.loss_file]

    def delete(self, *args, **kwargs):
        for file in self.files():
            if file:
                file.delete()
        super().delete(*args, **kwargs)

    def run(self):
        df = self.period.rate_file.df()
        df1 = pd.get_dummies(data=df, columns=['adviser_0'])
        for col in df1: df1[col].replace({True: 1, False: 0}, inplace=True)
        cols_to_scale = ['age_0', ]
        scaler = MinMaxScaler()
        df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
        x = df1.drop(['rate', 'cat_continuing', 'cat_exit'], axis='columns')
        y = df1['rate']

        details = string_to_array(self.layers)
        print("\nDetails:", details)
        layers = []
        for count, layer_size in enumerate(details):
            print("Layer Size:",layer_size)
            if count == 0: layer = Dense(layer_size, input_shape=(x.shape[1],), activation='relu')
            else:          layer = Dense(layer_size, activation='relu')
            layers.append(layer)
        layers.append(Dense(1, activation='sigmoid'))
        model = Sequential(layers)

        model.compile("adam", "binary_crossentropy", metrics=["MeanSquaredError"])
        history = model.fit(x, y, epochs=self.epochs)
        loss = pd.DataFrame(history.history['loss'], columns=['loss', ])
        loss = loss.iloc[::5]
        loss.index.name = "epoch"
        self.loss_file = create_file("loss", self.name, loss, index=True)
        self.mse = round(model.evaluate(x, y)[1], 4)
        print(f"Final MSE: {self.mse}")

        # Save the model
        filename = f"models/{self.name}.tf"
        self.model = filename
        self.save()
        if os.path.isfile(filename) is False:
            model.save(filename)
        prediction = model.predict(x)

        # Create the assumption table
        df['rate'] = prediction
        value = df.apply(self.apply_value, table=df, axis=1)
        df = pd.concat([df, value], axis=1)
        df.rename({0: "value"}, inplace=True, axis=1)

        self.assumption_file = create_file("assumption", self.name, df)
        self.save()

    def apply_value(self, data, table):
        age_s = data['age_0']
        adviser = data['adviser_0']

        # Time
        time = np.arange(self.proj_period)
        age = age_s + time
        time = pd.Series(time)
        age = pd.Series(age)
        discount_rate = 0.10
        inv_earnings = 0.07

        df = pd.concat([time, age], axis=1)
        df.rename({0: "time", 1: "age"}, inplace=True, axis=1)
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

    # def create_table(self):
    #     if not self.model: return
    #     model = load_model(self.model)
    #     df_encoded = self.period.encoded_rate_file.df()
    #     df = self.period.rate_file.df()
    #     x = df_encoded.drop(['rate', 'cat_continuing', 'cat_exit'], axis='columns')
    #     print("df_encoded\n", df_encoded)
    #     print("x\n", x)
    #
    #     predictions = model.predict(x)
    #     print("Predictions\n", predictions)
    #     prediction_list = []
    #     for prediction in predictions: prediction_list.append(prediction[0])
    #     df['rate'] = prediction_list
    #     self.assumption_file = create_file("assumption", self.name, df)
    #     self.save()

    # def create_table_old(self):
    #     if not self.model: return
    #     df = self.period.encoded_rate_file.df()
    #     df = df.drop('rate', axis='columns')
    #     model = load_model(self.model)
    #     table = []
    #     for age in AGE_RANGE:
    #         template = {col: 0 for col in df.columns}
    #         template['age_0'] = age
    #         for adviser in ADVISER_RANGE:
    #             sample = template
    #             adviser_column = 'adviser_0_' + adviser
    #             sample[adviser_column] = 1
    #             print(sample)
    #             input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    #             prediction = round(model.predict(input_dict)[0][0], 3)
    #             row = (age, adviser, prediction)
    #             table.append(row)
    #     df = pd.DataFrame(data=table, index=None, columns=['age_0', 'adviser_0', 'rate'])
    #     self.assumption_file = create_file("assumption", self.name, df)
    #     self.save()
    #
    #
    # def run_old(self):
    #     train_dataframe = self.period.rate_file.df()
    #     train_ds = dataframe_to_dataset(train_dataframe, "rate")
    #     train_ds = train_ds.batch(32)
    #
    #     # Encode the data
    #     age = Input(shape=(1,), name="age_0")
    #     adviser = Input(shape=(1,), name="adviser_0")
    #     age_encoded = encode_numerical_feature(age, "age_0", train_ds)
    #     adviser_encoded = encode_categorical_feature(adviser, "adviser_0", train_ds, is_string=True)
    #
    #     print("\nage:\n", age)
    #     print("\nage encoded:\n", age_encoded)
    #     print("\nadviser:\n", adviser)
    #     print("\nadviser encoded:\n", adviser_encoded)
    #
    #
    #     all_inputs = [age, adviser,]
    #     all_features = concatenate([age_encoded, adviser_encoded,])
    #
    #     # Define the  model
    #     x = Dense(32, activation="relu")(all_features)
    #     x = Dense(32, activation="relu")(x)
    #     x = Dense(32, activation="relu")(x)
    #     output = Dense(1, activation="sigmoid")(x)
    #     model = KerasModel(all_inputs, output)
    #
    #     # Compile and run the model
    #     model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    #     model.fit(train_ds, epochs=self.epochs)
    #
    #     # Save the model
    #     print(model.summary())
    #     filename = f"models/{self.name}.tf"
    #     self.model = filename
    #     self.save()
    #     if os.path.isfile(filename) is False:
    #         model.save(filename)
    #     self.create_table()

    # def plot_in(self):
    #     try:
    #         df = self.period.rate_file.df()
    #         return self.plot(df)
    #     except:
    #         return
    #
    # def plot_out(self):
    #     try:
    #         df = self.assumption_file.df()
    #         return self.plot(df)
    #     except:
    #         return
    #
    # def plot(self, df):
    #     if df is None: return
    #
    #     fig, ax = plt.subplots()
    #     for adviser_0, group_df in df.groupby('adviser_0'):
    #         plt.plot(group_df['age_0'], group_df['rate'], label=f"Adviser {adviser_0}")
    #
    #     # Customize the plot
    #     plt.title("Rate vs. Age by Adviser")
    #     plt.xlabel("Age")
    #     plt.ylabel("Rate")
    #     plt.legend()
    #     plt.grid(False)
    #
    #     html = mpld3.fig_to_html(fig)
    #     return html

    def plot(self):
        if not self.period.rate_file: return
        if not self.assumption_file: return

        df_in = self.period.rate_file.df()
        df_out = self.assumption_file.df()
        if df_in is None or df_out is None: return

        fig, ax = plt.subplots()
        colours = ['green', 'blue', 'red']
        count = 0
        for adviser_0, group_df in df_in.groupby('adviser_0'):
            plt.plot(group_df['age_0'], group_df['rate'], label=f"Adviser {adviser_0}", color=colours[count])
            count += 1
        count = 0
        for adviser_0, group_df in df_out.groupby('adviser_0'):
            plt.plot(group_df['age_0'], group_df['rate'], label=f"Adviser {adviser_0}", color=colours[count], linestyle='dotted')
            count += 1

        # Customize the plot
        plt.title("Rate vs. Age by Adviser")
        plt.xlabel("Age")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(False)

        html = mpld3.fig_to_html(fig)
        return html

    def loss(self):
        if not self.loss_file: return None
        df = self.loss_file.df()
        final = df['loss'].iloc[-1]
        try:
            final50 = df['loss'].iloc[-11]
            improvement = round(100-(final50/final) * 100, 3)
            final = round (final, 2)

            return f"{final} ({improvement}%)"
        except:
            final = round (final, 2)
            return f"{final}"

    def plot_loss(self):
        if not self.loss_file: return

        df = self.loss_file.df()
        if df is None: return

        loss = round(df['loss'].iloc[-1], 2)

        fig, ax = plt.subplots()
        plt.plot(df['epoch'], df['loss'], label="Loss", color='blue')

        # Customize the plot
        plt.title(f"Loss (Min={loss})", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.legend()
        plt.grid(False)

        html = mpld3.fig_to_html(fig)
        return html

def get_category(info):
    age_s_0 = info['age_0']
    age_s_1 = info['age_1']
    if math.isnan(age_s_0): return "new"
    if math.isnan(age_s_1): return "exit"
    return "continuing"


time_proj_1 = None
db = None
proj_count = 0
class Projection(Model):
    name = CharField(max_length=60, null=True)
    start = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="start_data")
    end = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="end_data")
    dnn = ForeignKey(DNN, blank=True, null=True, on_delete=SET_NULL, related_name="dnn_model")
    end_date = DateField(null=True, blank=True)
    proj_period = IntegerField(blank=True, null=True, default=5)
    file = ForeignKey(FileModel, related_name="projection_file", on_delete=SET_NULL, null=True)
    file_time_proj = ForeignKey(FileModel, related_name="time_proj", on_delete=SET_NULL, null=True)
    file_db = ForeignKey(FileModel, related_name="db", on_delete=SET_NULL, null=True)
    input_df = None
    assumption_df = None
    output_df = None

    def __str__(self):
        if self.name: return self.name
        else: "No name given"

    def files(self):
        return self.file, self.file_time_proj, self.file_db

    def delete(self, *args, **kwargs):
        for file in self.files():
            if file:
                file.delete()
        super().delete(*args, **kwargs)

    def create_input_df(self):
        try:
            df_s = pd.DataFrame(data=self.start.file.df(), index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
            df_e = pd.DataFrame(data=self.end.file.df(), index=None, columns=['number', 'age', 'adviser', 'fum_s', 'dur_s'])
            df = df_s.join(df_e.set_index('number'), on='number', how='outer', lsuffix='_0', rsuffix='_1')
            df = df[~df.index.duplicated(keep='first')]
            df['adviser_0'] = df['adviser_0'].combine_first(df['adviser_1'])
            return df
        except:
            return "Dataframe failure"

    def run(self):
        # Get data
        self.input_df = self.create_input_df()
        # self.save()
        # Run
        output_df, time_proj, db_file = self.run_df(self.input_df)
        # Save results
        self.file = create_file("projection", self.name, output_df)
        self.file_time_proj = time_proj
        self.file_db = db_file
        self.save()

    def run_df(self, df, proj_ind=None):
        global time_proj_1
        # Get assumptions
        if not self.assumption_df:
            self.assumption_df = self.dnn.assumption_file.df()
            self.save()

        # Categorise each row
        category = df.apply(get_category, axis=1)
        input_df = pd.concat([df, category], axis=1)
        input_df.rename(columns={0: "cat", }, inplace=True)

        # Add calculated variables
        output_df = input_df.apply(self.run_policy_aoc, proj_ind=proj_ind, axis=1)
        output_df = pd.concat([input_df, output_df], axis=1)
        time_proj_1['Time'] = np.arange(self.proj_period)
        time_proj = create_file("time proj", self.name, time_proj_1)
        db_file = create_file("proj_db", self.name, db)
        return output_df, time_proj, db_file

    def run_policy_aoc(self, info, proj_ind=None):
        info_0 = info[1:5]
        info_1 = info[5:9]
        time_proj_df = None

        output_0 = self.run_policy(info_0, time_period=0, proj_ind=proj_ind)
        output_1 = self.run_policy(info_1, time_period=1, proj_ind=proj_ind)

        output_0.rename({0: "Start", 1: "Less Profit", 2: "Rollforward", 3: "Expected"}, inplace=True)
        output_1.rename({0: "End"}, inplace=True)
        output_d = pd.Series([output_1["End"] - output_0["Expected"], f"http://127.0.0.1:8000/show/{self.id}/{info.iloc[0]}"])
        output_d.rename({0: "Delta", 1: "Link"}, inplace=True)
        output = pd.concat([output_0, output_1, output_d])

        return output

    def run_policy(self, data, time_period, proj_ind=None):
        global time_proj_1, proj_count, db
        # Data (for reading a series)
        age_s = data[f'age_{time_period}']
        fum_s = data[f'fum_s_{time_period}']
        adviser = data[f'adviser_{time_period}']

        if math.isnan(age_s):
            if time_period == 0: return pd.Series([0, 0, 0, 0])
            else:                return pd.Series([0, ])

        # Time
        time = np.arange(self.proj_period)
        age = age_s + time

        # Economic Assumptions
        discount_rate = 0.10
        inv_earnings = 0.07
        disc_factor = (1 / (1 + discount_rate)) ** time

        # Policy Assumptions
        lapse_rate = self.calc_lapse_rate_series(self.assumption_df, age, adviser)

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

        time = pd.Series(time)
        age = pd.Series(age)
        adviser = pd.Series([adviser] * self.proj_period)
        policies = pd.Series(policies)
        fum = pd.Series(fum)
        profit = pd.Series(profit)
        time_proj_ind = pd.concat([time, lapse_rate, policies, fum, profit], axis=1)
        time_proj_ind.rename(columns={0: "Time", 1: "Lapse Rate", 2: "Policies", 3: "FUM", 4: "Profit"}, inplace=True)
        db_ind = pd.concat([time, age, adviser, lapse_rate, policies, fum, profit], axis=1)
        db_ind.rename(columns={0: "Time", 1: "Age", 2: "Adviser", 3: "Lapse Rate", 4: "Policies", 5: "FUM", 6: "Profit"}, inplace=True)

        if time_period == 0:
            output = pd.Series([pv_profit_0,-profit[0],roll_forward,pv_profit_1_e])
        else:
            output = pd.Series([pv_profit_0,])
            if time_proj_1 is None:
                time_proj_1 = time_proj_ind
                db = db_ind
            else:
                time_proj_1 = time_proj_1.add(time_proj_ind, fill_value=0)
                db = pd.concat([db, db_ind], axis=0)

        if proj_ind:
            name = f"{proj_ind.projection}_{proj_ind.number}_{time_period}"
            if time_period == 0:
                proj_ind.file_start = create_file("proj_ind", name, time_proj_ind)
            else:
                proj_ind.file_end = create_file("proj_ind", name, time_proj_ind)
            proj_ind.save()
            # detail.to_csv(f"temp/projection_ind_{time_period}.csv")
        # else:
        #     print("No proj ind")
        return output

    def calc_lapse_rate_series(self, table, ages, adviser):
        result = []
        for age in ages:
            row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
            print(f"Calc lapse rate: {age}\n", row)
            try:
                rate = row.iloc[0]['rate']
            except:
                rate = 1
            result.append(rate)
        result = pd.Series(result)
        return result

class Proj_Ind(Model):
    name = CharField(max_length=60, null=True)
    number = IntegerField(null=True)
    projection = ForeignKey(Projection, blank=True, null=True, on_delete=SET_NULL, related_name="proj_ind")
    file = ForeignKey(FileModel, related_name="proj_ind_file", on_delete=SET_NULL, null=True)
    file_start = ForeignKey(FileModel, related_name="proj_ind_start", on_delete=SET_NULL, null=True)
    file_end = ForeignKey(FileModel, related_name="proj_ind_end", on_delete=SET_NULL, null=True)
    def __str__(self): return f"{self.projection}: Record {self.number}"

    def input_df(self):
        query_string = f"number == {self.number}"
        input_df = self.projection.get_input_df().query(query_string)
        return input_df

    def input_df_html(self):
        return self.input_df().to_html(classes=['table', 'table-striped', 'table-center'], index=False, justify='center', formatters=formatters, render_links=True,)

    def run(self):
        # Run
        output_df, time_proj, db_file = self.projection.run_df(self.input_df(), self)
        # Save results
        self.file = create_file("proj_ind", f"{self.projection.name}_{self.number}", output_df)
        self.save()

class Setting(Model):
    name = CharField(max_length=512, null=True)
    projection = ForeignKey(Projection, on_delete=SET_NULL, null=True)
    def __str__(self):
        if self.name: return self.name
        else: "No name given"

all_models = [Setting, FileModel, Data, Join, Period, DNN, Projection, Proj_Ind]

