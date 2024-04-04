from django.shortcuts import render, redirect
from app.a1_data import *
from app.old.a4_projection import *
from .forms import *

def home(request):
    # lapse_analyses = LapseAnalysis.objects.all()
    context = {}
    return render(request, "home.html", context)

def list(request, model_str):
    model, form = get_model(model_str)
    items = model.objects.all()
    types = ['data', 'join', 'period', 'rate', 'assumption', 'projection', 'proj_ind']
    context = {'items': items, 'model_str': model_str, 'types': types}
    return render(request, model_str + "s.html", context)

def item(request, model_str, id):
    model, form = get_model(model_str)
    item = model.objects.get(id=id)
    context = {'item': item, 'model_str': model_str,}
    return render(request, model_str + ".html", context)

def new(request, model_str):
    model, form = get_model(model_str)
    if request.method == 'POST':
        form = form(request.POST)
        if form.is_valid():
            new = form.save()
            if model_str == "period":
                new.create_file()
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
    elif model_str in ["data", "join", "period", "projection",]:
        if item.file:
            item.file.delete()
    if model_str == "data":
        Join.objects.filter(start=item).delete()
        Join.objects.filter(end=item).delete()
    item.delete()
    return redirect("list", model_str)

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

def data_create(request):
    create_first_data()
    return redirect("list", "data")

def data_create_next(request, id):
    data = Data.objects.get(id=id)
    data.create_next()
    return redirect("list", "data")

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

