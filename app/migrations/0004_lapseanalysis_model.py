# Generated by Django 4.2.9 on 2024-02-01 00:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_alter_lapseanalysis_end_alter_lapseanalysis_output_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='lapseanalysis',
            name='model',
            field=models.CharField(max_length=512, null=True),
        ),
    ]
