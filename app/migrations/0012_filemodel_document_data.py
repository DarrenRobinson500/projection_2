# Generated by Django 4.2.9 on 2024-03-28 05:29

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_general_detailed_output_alter_filemodel_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='filemodel',
            name='document',
            field=models.FileField(blank=True, null=True, upload_to='files/'),
        ),
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=512, null=True)),
                ('date', models.DateField(blank=True, null=True)),
                ('file', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='data_file', to='app.filemodel')),
            ],
        ),
    ]
