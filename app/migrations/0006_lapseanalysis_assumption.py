# Generated by Django 4.2.9 on 2024-02-01 11:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_lapseanalysis_epochs'),
    ]

    operations = [
        migrations.AddField(
            model_name='lapseanalysis',
            name='assumption',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='assumption', to='app.filemodel'),
        ),
    ]