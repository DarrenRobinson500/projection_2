# Generated by Django 4.2.9 on 2024-04-03 04:15

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0023_data_index'),
    ]

    operations = [
        migrations.CreateModel(
            name='Proj_Ind',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=60, null=True)),
                ('number', models.IntegerField(null=True)),
                ('file', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='proj_ind_file', to='app.filemodel')),
                ('projection', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='proj_ind', to='app.projection')),
            ],
        ),
    ]
