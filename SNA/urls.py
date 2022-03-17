from django.urls import path

from . import views


# 将view的调用映射到url
urlpatterns = [
    path('', views.index, name='index'),
    # 数据集相关视图
    path('<int:dataset_id>/', views.dataset_detail, name='d_detail'),
    path('<int:dataset_id>/upload/', views.dataset_upload, name='d_upload'),
]
