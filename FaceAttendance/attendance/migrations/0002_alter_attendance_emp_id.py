# Generated by Django 5.1.6 on 2025-02-21 18:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendance',
            name='emp_id',
            field=models.IntegerField(null=True),
        ),
    ]
