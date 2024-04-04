# Generated by Django 4.2.9 on 2024-01-26 03:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='lapseanalysis',
            name='experience',
        ),
        migrations.AddField(
            model_name='lapseanalysis',
            name='rate',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='rate', to='app.filemodel'),
        ),
        migrations.AlterField(
            model_name='filemodel',
            name='type',
            field=models.CharField(blank=True, choices=[('data', 'Data File'), ('lapse', 'Lapse File'), ('rate', 'Rate File')], max_length=100, null=True),
        ),
    ]