from django.forms import *
from .models import *

class PeriodForm(ModelForm):
    class Meta:
        model = Period
        fields = ['name', 'start_date', 'end_date', ]
        widgets = {
            'start_date': DateInput(attrs={'class': 'form-control', 'type':'date'}),
            'end_date': DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'name': TextInput(attrs={'class': 'form-control', }),
        }

class DNNForm(ModelForm):
    class Meta:
        model = DNN
        fields = ['name', 'period', 'epochs', ]
        widgets = {
            'name': TextInput(attrs={'class': 'form-control', }),
            'period': Select(attrs={'class': 'form-control', }),
            'epochs': TextInput(attrs={'class': 'form-control', }),
        }

class ProjectionForm(ModelForm):
    class Meta:
        model = Projection
        fields = ['name', 'start', 'end', 'dnn', ]
        widgets = {
            'name': TextInput(attrs={'class': 'form-control', }),
            'start': Select(attrs={'class': 'form-control', }),
            'end': Select(attrs={'class': 'form-control', }),
            'dnn': Select(attrs={'class': 'form-control', }),
        }

class SettingForm(ModelForm):
    class Meta:
        model = Setting
        fields = ['name', 'projection']
        widgets = {
            'name': TextInput(attrs={'class': 'form-control',}),
            'projection': Select(attrs={'class': 'form-control', }),
        }

def get_model(model_str):
    if model_str == "file": return FileModel, None
    if model_str == "data": return Data, None
    if model_str == "join": return Join, None
    if model_str == "period": return Period, PeriodForm
    if model_str.lower() == "dnn": return DNN, DNNForm
    if model_str == "projection": return Projection, ProjectionForm
    if model_str == "proj_ind": return Proj_Ind, None
    if model_str == "setting": return Setting, SettingForm




# class LapseForm(Form):
#     name = CharField(widget=TextInput(attrs={'class': 'form-control'}))
#     file_start = ModelChoiceField(queryset=FileModel.objects.all(), widget=Select(attrs={'class': 'form-control'}))
#     file_end = ModelChoiceField(queryset=FileModel.objects.all(), widget=Select(attrs={'class': 'form-control'}))
    # file_output = CharField(widget=TextInput(attrs={'class': 'form-control'}))
    # file_output = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))

    # comment = forms.CharField(widget=forms.Textarea)

# class LapseModelForm(ModelForm):
#     class Meta:
#         model = LapseAnalysis
#         fields = ['name', 'epochs', ]
#         widgets = {
#             'name': TextInput(attrs={'class':'form-control', 'placeholder': "Name"}),
#             'epochs': TextInput(attrs={'class':'form-control', 'placeholder': "Epochs"}),
#         }
#
#
