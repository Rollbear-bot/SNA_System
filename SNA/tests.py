from django.test import TestCase

# Create your tests here.
# 此处为测试模块

import datetime

from django.test import TestCase
from django.utils import timezone

from .models import Dataset


# test a model
class DatasetModelTests(TestCase):

    def test_something(self):
        # assert false is false
        # app test ay "python manage.py test polls"
        self.assertIs(False, False)


# test a view

