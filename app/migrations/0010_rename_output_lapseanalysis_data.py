# Generated by Django 4.2.9 on 2024-02-02 20:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0009_general_name'),
    ]

    operations = [
        migrations.RenameField(
            model_name='lapseanalysis',
            old_name='output',
            new_name='data',
        ),
    ]