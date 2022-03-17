from django.db import models

# Create your models here.
# A model is the single, definitive source of information about your data.
# It contains the essential fields and behaviors of the data you’re storing.
# Django follows the DRY Principle. The goal is to define your data model in one place
# and automatically derive things from it.


class User(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return f"User '{self.name}'"


class Dataset(models.Model):
    """
    数据模型：数据集
    """
    # 数据集名称
    name = models.CharField(max_length=200)
    # 数据集（文件形式）在后端的存储路径
    path = models.CharField(max_length=200)
    # 用户上传该数据集的日期时间
    pub_date = models.DateTimeField('date published')  # 可定义human-readable name
    # 上传者（此处是否应该使用外键？）
    # PROTECT关系表示有数据集存在时，上传者（用户）不能被删除
    owner = models.ForeignKey(User, on_delete=models.PROTECT)

    def __str__(self):
        return f"{self.name}, uploaded at {self.pub_date}, by {self.owner}"
