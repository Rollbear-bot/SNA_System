from django.shortcuts import render
from django.http import HttpResponse
from .models import Dataset
from django.template import loader

# Create your views here.
# view要么返回HttpResponse，要么返回HttpException


def index(req):
    """默认转发页面，展示最多5个最新的数据集
    点击数据集名可以跳转到数据集详情页"""
    # return HttpResponse("hello!")
    lasted_pub_lt = Dataset.objects.order_by("pub_date")[:5]
    template = loader.get_template("SNA/index.html")
    # 关联HTML的内容
    context = {
        "dataset_lt": lasted_pub_lt
    }
    return HttpResponse(
        template.render(context, req))


def dataset_detail(req, dataset_id):
    return HttpResponse(f"You're looking at dataset {dataset_id}.")


def dataset_upload(req, dataset_id):
    return HttpResponse(f"visit here to upload a dataset.")

