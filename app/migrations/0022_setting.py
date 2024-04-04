# Generated by Django 4.2.9 on 2024-03-31 22:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0021_remove_lapseanalysis_assumption_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Setting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=512, null=True)),
                ('projection', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='app.projection')),
            ],
        ),
    ]