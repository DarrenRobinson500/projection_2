from django.contrib import admin
from django.urls import path
from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", home, name="home"),
    path("home", home, name="home"),
    path("list/<model_str>", list, name="list"),
    path("item/<model_str>/<id>", item, name="item"),
    path("new/<model_str>", new, name="new"),
    path("edit/<model_str>/<id>", edit, name="edit"),
    path("delete/<model_str>/<id>", delete, name="delete"),
    path("file/<model_str>/<id>", file, name="file"),
    path("run/<model_str>/<id>", run, name="run"),
    path("show/<projection_id>/<number>", show, name="show"),
    path("show/<projection_id>/<number>/<force_recalc>", show, name="show"),

    # path("data", data, name="data"),
    # path("data_ind/<id>", data_ind, name="data_ind"),
    path("data_create", data_create, name="data_create"),
    path("data_create_next/<id>", data_create_next, name="data_create_next"),

    # path("joins", joins, name="joins"),
    # path("join/<id>", join, name="join"),
    #
    # path("periods", periods, name="periods"),
    # path("period_create", period_create, name="period_create"),
    # path("period/<id>", period, name="period"),
    #
    # path("lapse", lapse, name="lapse"),
    # path("lapse_run", lapse_run, name="lapse_run"),
    # path("lapse_outcome/<id>", lapse_outcome, name="lapse_outcome"),
    # path("models", models, name="models"),
    # path("model_run/<id>", model_run, name="model_run"),
    # path("model_ind/<id>", model_ind, name="model_ind"),
    # path("projections", projections, name="projections"),
    # path("projection_run/<id>", projection_run, name="projection_run"),
]
