from django.urls import path

from . import views

app_name = "SNA"

# 将view的调用映射到url
urlpatterns = [
    path('', views.index, name='index'),
    # 数据集相关视图
    path('<int:dataset_id>/', views.dataset_detail, name='d_detail'),
    path('upload/', views.dataset_upload, name='d_upload'),
    path('run_alg/', views.run_alg, name='run_alg'),
    path('register/', views.Register.as_view(), name='register'),
    path('login/', views.Login.as_view(), name='login')
]
