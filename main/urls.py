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
    path("delete_all/<model_str>", delete_all, name="delete_all"),
    path("file/<model_str>/<id>", file, name="file"),
    path("run/<model_str>/<id>", run, name="run"),
    path("create_files/<model_str>/<id>", create_files, name="create_files"),
    path("create_table/<model_str>/<id>", create_table, name="create_table"),

    path("data_create", data_create, name="data_create"),
    path("data_create_next/<id>", data_create_next, name="data_create_next"),
    path("show/<projection_id>/<number>", show, name="show"),
    path("show/<projection_id>/<number>/<force_recalc>", show, name="show"),
    path("proj_analysis/<id>", proj_analysis, name="proj_analysis"),
]
