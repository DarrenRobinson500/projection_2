from keras.models import load_model
from .utilities import *
from keras import Input
from keras import Model as KerasModel
import matplotlib.pyplot as plt
import mpld3
import math, random

import os
import pandas as pd
from django.db.models import *
from django.core.files import File

def format_2dp(var): return f'{var:,.2f}'
def format_age(var): return f'{var:.0f}'
def format_int(var): return f'{var:.0f}'
def format_adviser(var): return f'{var:.0f}'
def format_dollars(var): return f'{var:,.0f}'

formatters = {
    'Time': format_int,
    'Policies': format_2dp,
    'FUM': format_dollars, 'fum_s': format_dollars, 'fum_s_0': format_dollars, 'fum_s_1': format_dollars,
    'Profit': format_2dp,
    'dur_s_0': format_int, 'dur_s_1': format_int,
    'rate': format_2dp,
    'age_0': format_int, 'age_1': format_int,
    'adviser_0': format_int,'adviser_1': format_int,
    'Start': format_dollars, 'Less Profit': format_dollars, 'Rollforward': format_dollars,
    'Expected': format_dollars, 'End': format_dollars, 'Delta': format_dollars,
}

def create_file(type, name, df):
    file_name = f"files/{type}/{name}.csv"

    directory = f"files/{type}"
    if not os.path.exists(directory): os.makedirs(directory)

    existing_files = FileModel.objects.filter(name=file_name)
    existing_files.delete()

    df.to_csv(file_name, index=False, header=True)
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
    name = CharField(max_length=512)
    time_stamp = DateTimeField(auto_now_add=True, null=True,blank=True)
    last_update = DateTimeField(null=True,blank=True)
    document = FileField(upload_to="files/", blank=True, null=True)
    type = CharField(max_length=100, blank=True, null=True, choices=TYPE_CHOICES)

    def __str__(self):
        if self.name: return self.name
        else: "No name given"

        return self.name

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
        html += f"<h3>All Records</h3>"
        # html.replace(">http://127.0.0.1:8000/show/", "Show ")

        return html

    def df_top(self):
        df = pd.read_csv(str(self.name), nrows=4)
        return df

    def df_top_html(self):
        return self.df_top().to_html(classes=['table', 'table-striped', 'table-center'], index=False, justify='center', formatters=formatters, render_links=True,)

    def delete(self, *args, **kwargs):
        self.document.delete()
        super().delete(*args, **kwargs)

def create_first_data():
    data = []
    fum = 10000
    dur = 5
    i = 1
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            for x in range(10):
                data.append((i, age, adviser, fum, dur))
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
    name = CharField(max_length=512, null=True)
    date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="data_file", on_delete=SET_NULL, null=True)
    index = IntegerField(null=True, blank=True)
    def __str__(self):
        if self.name: return self.name
        else: "No name given"

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
        table = projection.dnn.assumption_file.df()
        row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
        annual_rate = row.iloc[0]['rate']
        rate = 1 - (1 - annual_rate) ** (1/12)
        random_number = random.random()
        if random_number < rate: return "Delete"
        return "Keep"

    def add_new_business(self, df):
        if not self.index:
            index = 1
        else:
            index = self.index + 1
        fum, duration = 100000, 0
        for age in AGE_RANGE_NEW:
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


class Join(Model):
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
        if self.start is None or self.start.file.df() is None: return None
        if self.end is None or self.end.file.df() is None: return None

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
        self.file = create_file("join", f"{self}.csv", self.df())
        self.end_date = self.end.date
        self.save()

class Period(Model):
    name = CharField(max_length=10, null=True)
    start_date = DateField(null=True, blank=True)
    end_date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="period_file", on_delete=SET_NULL, null=True)
    rate_file = ForeignKey(FileModel, related_name="rate_file", on_delete=SET_NULL, null=True)
    def __str__(self): return f"{self.start_date} to {self.end_date}"

    def joins(self):
        return Join.objects.filter(end_date__range=[self.start_date, self.end_date])

    def df(self):
        joins = self.joins()
        for count, join in enumerate(joins):
            if count == 0: df = join.df()
            else:
                df = pd.concat([df, join.df()], ignore_index=True)
        return df

    def create_rate_file(self):
        grouped = self.df().groupby(["age_0", "adviser_0"]).count()
        survivors = grouped["age_1"]
        start = grouped["number"]
        rate = (1 - survivors / start)
        rate_df = rate.to_frame()
        rate_df.columns = ['rate', ]
        rate_df.to_csv("files/rate.csv", index=True, header=True)
        rate_df = pd.read_csv("files/rate.csv")

        self.rate_file = create_file("rate", f"{self}.csv", rate_df)
        self.save()

    def create_file(self):
        self.file = create_file("period", f"{self}.csv", self.df())
        self.save()
        self.create_rate_file()

class DNN(Model):
    name = CharField(max_length=30, null=True)
    period = ForeignKey(Period, related_name="period", on_delete=SET_NULL, null=True)
    epochs = IntegerField(default=50)
    assumption_file = ForeignKey(FileModel, related_name="assumption", on_delete=SET_NULL, null=True)
    model = CharField(max_length=512, null=True)
    def __str__(self): return self.name

    def run(self):
        train_dataframe = self.period.rate_file.df()
        train_ds = dataframe_to_dataset(train_dataframe, "rate")
        train_ds = train_ds.batch(32)

        # Encode the data
        age = Input(shape=(1,), name="age_0")
        adviser = Input(shape=(1,), name="adviser_0", dtype="int64")
        age_encoded = encode_numerical_feature(age, "age_0", train_ds)
        adviser_encoded = encode_categorical_feature(adviser, "adviser_0", train_ds, False)

        all_inputs = [age, adviser,]
        all_features = concatenate([age_encoded, adviser_encoded,])

        # Define the  model
        x = Dense(32, activation="relu")(all_features)
        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)
        model = KerasModel(all_inputs, output)

        # Compile and run the model
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, epochs=self.epochs)

        # Save the model
        print(model.summary())
        filename = f"models/{self.name}.tf"
        self.model = filename
        self.save()
        if os.path.isfile(filename) is False:
            model.save(filename)
        self.create_table()

    def create_table(self):
        if not self.model: return
        model = load_model(self.model)

        AGE_RANGE = range(20,66)
        ADVISER_RANGE = [1, 2, 3]

        table = []
        for age in AGE_RANGE:
            for adviser in ADVISER_RANGE:
                sample = {"age_0": age, "adviser_0": adviser, }
                input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
                prediction = round(model.predict(input_dict)[0][0], 3)
                row = (age, adviser, prediction)
                table.append(row)
        df = pd.DataFrame(data=table, index=None, columns=['age_0', 'adviser_0', 'rate'])
        self.assumption_file = create_file("assumption", self.name, df)
        self.save()

    def plot_in(self):
        df = self.period.rate_file.df()
        return self.plot(df)

    def plot_out(self):
        df = self.assumption_file.df()
        return self.plot(df)

    def plot(self, df):
        if df is None: return

        fig, ax = plt.subplots()
        for adviser_0, group_df in df.groupby('adviser_0'):
            plt.plot(group_df['age_0'], group_df['rate'], label=f"Adviser {adviser_0}")

        # Customize the plot
        plt.title("Rate vs. Age by Adviser")
        plt.xlabel("Age")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(False)

        html = mpld3.fig_to_html(fig)
        return html

def get_category(info):
    age_s_0 = info['age_0']
    age_s_1 = info['age_1']
    if math.isnan(age_s_0): return "new"
    if math.isnan(age_s_1): return "exit"
    return "continuing"

class Projection(Model):
    name = CharField(max_length=60, null=True)
    start = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="start_data")
    end = ForeignKey(Data, blank=True, null=True, on_delete=SET_NULL, related_name="end_data")
    dnn = ForeignKey(DNN, blank=True, null=True, on_delete=SET_NULL, related_name="dnn_model")
    end_date = DateField(null=True, blank=True)
    file = ForeignKey(FileModel, related_name="projection_file", on_delete=SET_NULL, null=True)
    input_df = None
    assumption_df = None
    output_df = None
    def __str__(self):
        if self.name: return self.name
        else: "No name given"

    def get_input_df(self):
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
        self.input_df = self.get_input_df()
        self.save()
        # Run
        output_df = self.run_df(self.input_df)
        # Save results
        self.file = create_file("projection", self.name, output_df)
        self.save()

    def run_df(self, df, proj_ind=None):
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
        return output_df

    def run_policy_aoc(self, info, proj_ind=None):
        info_0 = info[1:5]
        info_1 = info[5:9]

        output_0, detail_0 = self.run_policy(info_0, time_period=0, proj_ind=proj_ind)
        output_1, detail_1 = self.run_policy(info_1, time_period=1, proj_ind=proj_ind)

        output_0.rename({0: "Start", 1: "Less Profit", 2: "Rollforward", 3: "Expected"}, inplace=True)
        output_1.rename({0: "End"}, inplace=True)
        output_d = pd.Series([output_1["End"] - output_0["Expected"], f"http://127.0.0.1:8000/show/{self.id}/{info[0]}"])
        output_d.rename({0: "Delta", 1: "Link"}, inplace=True)
        output = pd.concat([output_0, output_1, output_d])
        self.acc_detail = detail_1

        return output

    def run_policy(self, data, time_period, proj_ind=None):
        # Data (for reading a series)
        age_s = data[f'age_{time_period}']
        fum_s = data[f'fum_s_{time_period}']
        adviser = data[f'adviser_{time_period}']

        if math.isnan(age_s):
            if time_period == 0: return pd.Series([0, 0, 0, 0])
            else:                return pd.Series([0, ])

        # Time
        time = np.arange(65-age_s)
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

        if time_period == 0:
            output = pd.Series([pv_profit_0,-profit[0],roll_forward,pv_profit_1_e])
        else:
            output = pd.Series([pv_profit_0,])

        time = pd.Series(time)
        policies = pd.Series(policies)
        fum = pd.Series(fum)
        profit = pd.Series(profit)
        detail = pd.concat([time, policies, fum, profit], axis=1)
        detail.rename(columns={0: "Time", 1: "Policies", 2: "FUM", 3: "Profit"}, inplace=True)

        if proj_ind:
            name = f"{proj_ind.projection}_{proj_ind.number}_{time_period}"
            if time_period == 0:
                proj_ind.file_start = create_file("proj_ind", name, detail)
                print("Created start\n", detail)
            else:
                proj_ind.file_end = create_file("proj_ind", name, detail)
                print("Created end\n", detail)
            proj_ind.save()
            # detail.to_csv(f"temp/projection_ind_{time_period}.csv")
        else:
            print("No proj ind")
        return output, detail

    def calc_lapse_rate_series(self, table, ages, adviser):
        result = []
        for age in ages:
            row = table.loc[(table['age_0'] == age) & (table['adviser_0'] == adviser)]
            rate = row.iloc[0]['rate']
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
        output_df = self.projection.run_df(self.input_df(), self)
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

