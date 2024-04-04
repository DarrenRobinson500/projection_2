# Generated by Django 4.2.9 on 2024-03-29 02:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0013_joint'),
    ]

    operations = [
        migrations.AddField(
            model_name='joint',
            name='file',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='joint_file', to='app.filemodel'),
        ),
        migrations.AlterField(
            model_name='filemodel',
            name='type',
            field=models.CharField(blank=True, choices=[('start', 'Start File'), ('end', 'End File'), ('data', 'Data File'), ('joint', 'Joint File'), ('lapse', 'Lapse File'), ('rate', 'Rate File'), ('assumption', 'Assumption File'), ('projection', 'Projection File')], max_length=100, null=True),
        ),
    ]