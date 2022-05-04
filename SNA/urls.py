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
    path('login/', views.Login.as_view(), name='login'),
    path('show_result/', views.tiles_plot, name='show_result'),
    path('dataset_alter/', views.dataset_alter, name='d_alter'),
    path('about', views.about_sys, name="about_sys"),
    path('d_upload_single', views.dataset_upload_single, name='d_upload_single'),
    path('d_lt', views.dataset_lt_single, name='d_lt_single'),
    path('comm_detail/<int:r_id>/<int:c_id>/<int:s_id>', views.comm_detail, name='c_detail'),
    path('comm_json_data/<int:r_id>/<int:c_id>/<int:s_id>/', views.get_comm_json_data, name='c_json'),
]
