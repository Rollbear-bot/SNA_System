from django.db import models
from django.contrib.auth.models import User


# Create your models here.
# A model is the single, definitive source of information about your data.
# It contains the essential fields and behaviors of the data you’re storing.
# Django follows the DRY Principle. The goal is to define your data model in one place
# and automatically derive things from it.


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
    # PROTECT关系表示有数据集存在时，上传者（用户）不能被删除，todo::添加用户信息
    owner = models.ForeignKey(User, on_delete=models.PROTECT, null=True)

    file = models.FileField(upload_to="./SNA/dataset_store/", null=True)

    # 公开or私有，默认值为私有
    is_private = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name}, uploaded at {self.pub_date}, by {self.owner}"


class RunResult(models.Model):
    """算法执行结果"""
    # 开始执行的时间
    exec_datetime = models.DateTimeField('date execute')
    # 外键链接到一个数据集
    dataset_used = models.ForeignKey(Dataset, on_delete=models.PROTECT)
    # 调用的算法名称
    alg_name = models.CharField(max_length=200)
    # 任务名
    task_name = models.CharField(max_length=200)
    # 执行用户：链接到一个用户
    exec_er = models.ForeignKey(User, on_delete=models.PROTECT)

    # 算法执行状态从tag文件中读取，不在数据库中保存
    # 执行结果不在数据库中保存

    def __str__(self):
        return f"on '{self.dataset_used.name}', start at {self.exec_datetime}"
