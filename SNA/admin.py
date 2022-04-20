from django.contrib import admin
from .models import Dataset, User, RunResult

# Register your models here.
# 在管理员界面中注册数据模型（使其可见/可操作）
admin.site.register(Dataset)
# admin.site.register(User)
admin.site.register(RunResult)
