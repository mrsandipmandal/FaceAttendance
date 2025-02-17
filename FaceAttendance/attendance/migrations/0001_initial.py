# Generated by Django 5.1.6 on 2025-02-17 17:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('emp_id', models.CharField(max_length=50, unique=True)),
                ('image', models.ImageField(upload_to='employees/')),
                ('face_encoding', models.BinaryField()),
            ],
        ),
    ]
