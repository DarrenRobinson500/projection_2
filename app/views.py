from django.shortcuts import render, redirect
from .forms import *

# -----------------------------
# ----- Generic Functions -----
# -----------------------------

def home(request):
    # lapse_analyses = LapseAnalysis.objects.all()
    all = []
    for model_str in ["data", "join", "period", "dnn", "projection", "file"]:
        model, form = get_model(model_str)
        items = model.objects.all()
        all.append((model_str, items))
    context = {"all": all}
    return render(request, "home.html", context)

def list(request, model_str):
    model, form = get_model(model_str)
    items = model.objects.all()
    dnns = DNN.objects.all()
    models = [Data, Join, Period, DNN]
    types = ['data', 'join', 'period', 'rate', 'assumption']
    context = {'items': items, 'model_str': model_str, 'models': models, 'types': types, "dnns": dnns, 'model': model}
    return render(request, model_str + "s.html", context)

def item(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    dnns = DNN.objects.all()
    context = {'item': item, 'model_str': model_str, "dnns": dnns}
    return render(request, model_str + ".html", context)

def new(request, model_str):
    model, form = get_model(model_str)
    if request.method == 'POST':
        form = form(request.POST)
        if form.is_valid():
            new = form.save()
            if model_str == "period": new.create_files()
            return redirect('list', model_str)
    form = form()
    context = {'form':form, 'model_str': model_str, 'mode': 'New'}
    return render(request, 'new.html', context)

def edit(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    if request.method == 'POST':
        form = form(request.POST or None, instance=item)
        if form.is_valid():
            new = form.save()
            if model_str == "period":
                new.create_file()
            return redirect('item', model_str, id)
    form = form(instance=item)
    context = {'form':form, 'model_str': model_str, 'mode': 'Edit'}
    return render(request, 'new.html', context)

def delete(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    if model_str == "file":
        if item.owner():
            item.owner().delete()
        else:
            item.delete()
    else:
        if item:
            item.delete()
    return redirect("list", model_str)

def delete_all(request, model_str):
    model, form = get_model(model_str)
    items = model.objects.all()
    for item in items:
        if model_str == "file":
            if item.owner():
                item.owner().delete()
            else:
                item.delete()
        else:
            if item:
                item.delete()
    return redirect("home")

def file(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    item.create_file()
    return redirect("item", model_str, id)

def run(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    item.run()
    return redirect("item", model_str, id)

def create_files(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    item.create_files()
    return redirect("item", model_str, id)

def create_table(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    item.create_table()
    return redirect("item", model_str, id)

# -----------------------------
# ----- Specific Functions -----
# -----------------------------

def data_create(request):
    create_first_data()
    return redirect("list", "data")

def data_create_next(request, id):
    data = Data.objects.get(id=id)
    data.create_next()
    return redirect("list", "data")

def apply_value(request, model_str, id, dnn_id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    dnn = DNN.objects.get(id=dnn_id)
    table = dnn.assumption_file.df()
    item.apply_value(table)
    return redirect("item", model_str, id)

def apply_values(request, model_str, dnn_id):
    model, form = get_model(model_str)
    items = model.objects.all()
    dnn = DNN.objects.get(id=dnn_id)
    table = dnn.assumption_file.df()
    for item in items:
        item.apply_value(table)
    return redirect('list', 'data')

def show(request, projection_id, number, force_recalc=False):
    projection = Projection.objects.get(id=projection_id)
    proj_ind = Proj_Ind.objects.filter(projection=projection, number=number).first()
    if force_recalc and proj_ind:
        proj_ind.delete()
    if not proj_ind:
        proj_ind = Proj_Ind(projection=projection, number=number)
        proj_ind.save()
    proj_ind.run()
    return redirect('item', "proj_ind", proj_ind.id)

def proj_analysis(request, id):
    item = Projection.objects.get(id=id)
    df = item.file_db.df()
    advisers = df['Adviser'].unique()

    sort_by = request.GET.get('sort_by')
    if sort_by: df.sort_values(by=sort_by, inplace=True)
    filter_on = request.GET.get('filter_on')
    if filter_on and filter_on in advisers: df = df[df['Adviser']==filter_on]
    options = ['Time', 'Age', 'Adviser', ]
    menu_options = ['Time', 'Age', 'Adviser', ]
    menu_data = []
    for option in menu_options:
        values = df[option].unique()
        menu_data.append((option, values))
    group_by = request.GET.get('group_by')

    groupings = menu_options
    if group_by:
        groupings.remove(group_by)
        df = df.groupby(groupings).sum()
        df = df.drop([group_by, ], axis='columns')
        # x = df1.drop(['rate', 'cat_continuing', 'cat_exit'], axis='columns')

    # df_count = df.groupby([group_by,]).sum()
    # df = df_sum
    # df['Time'] = df_count['Time']

    print("\nDataframe\n")
    print(type(df))
    print(df)

    shape = df.shape
    df = df.to_html(classes=['table', 'table-striped', 'table-center'], index=True, justify='center', formatters=formatters, render_links=True,)

    context = {'item': item, 'df': df, 'sort_options': options, 'advisers': advisers, 'shape': shape, 'menu_data': menu_data}
    return render(request, "proj_analysis.html", context)

