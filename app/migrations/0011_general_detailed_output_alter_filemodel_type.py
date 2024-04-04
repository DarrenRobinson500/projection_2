# Generated by Django 4.2.9 on 2024-02-04 05:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0010_rename_output_lapseanalysis_data'),
    ]

    operations = [
        migrations.AddField(
            model_name='general',
            name='detailed_output',
            field=models.BooleanField(null=True),
        ),
        migrations.AlterField(
            model_name='filemodel',
            name='type',
            field=models.CharField(blank=True, choices=[('start', 'Start File'), ('end', 'End File'), ('data', 'Data File'), ('lapse', 'Lapse File'), ('rate', 'Rate File'), ('assumption', 'Assumption File'), ('projection', 'Projection File')], max_length=100, null=True),
        ),
    ]
