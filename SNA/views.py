import json
import os

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse
from .models import Dataset, RunResult
from django.template import loader
from django.urls import reverse
from datetime import datetime, timedelta
from multiprocessing import Process
from SNA.algorithms.run_tiles import run_tiles
from SNA.algorithms.tiles_parser import *
from SNA.utils import get_file_size

# Create your views here.
# view要么返回HttpResponse，要么返回HttpException

# view可以通过generic的方式来继承
from django.views.generic import View
# django自带用户类
from django.contrib.auth.models import User
# django的用户功能库
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages

BASE_DIR = "SNA/dataset_store/"
SYS_META = {
    "SYS_VERSION": "v2.3.220503",
    "SYS_AUTHOR": "Junhong Wu",
    "SYS_YEAR": "2022",
    "SYS_NAME": "SNA System",
    "SYS_ZH_NAME": "社交网络动态社区发现与分析系统",
}


# 注册
class Register(View):

    def get(self, request):
        return render(request, 'SNA/register.html')

    def post(self, request):
        # 注册
        username = request.POST.get('username', '')  # 用户名
        password = request.POST.get('password', '')  # 密码
        check_password = request.POST.get('check_password', '')  # 确认密码

        # 检测密码与确认密码一致
        if password != check_password:
            messages.success(request, "密码不一致")
            return HttpResponseRedirect(reverse('SNA:register'))

        # 检测是否为空
        if username == '' or password == '' or check_password == '':
            messages.success(request, "不能为空!")
            return HttpResponseRedirect(reverse('SNA:register'))

        # 检测当前账号是否注册过并提示用户
        exists = User.objects.filter(username=username).exists()
        if exists:
            messages.success(request, "该账号已注册!")
            return HttpResponseRedirect(reverse('SNA:register'))
        User.objects.create_user(username=username, password=password)
        return HttpResponseRedirect(reverse('SNA:login'))


# 登录
class Login(View):

    def get(self, request):
        # 检测用户已登录过点击注册则跳转主页，注销才能重新登录，会保存15分钟登录状态
        # if request.user.is_authenticated:
        #     return HttpResponseRedirect(reverse('SNA:index'))
        return render(request, 'SNA/login.html')

    def post(self, request):
        # 登录
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')

        # 判断当前用户是否存在,不存在则重新注册
        exists = User.objects.filter(username=username).exists()
        if not exists:
            messages.success(request, "该账号不存在，请注册!")
            return HttpResponseRedirect(reverse('SNA:register'))

        # 检测是否为空
        # if username == '' or password == '' or check_password == '':
        #     messages.success(request, "不能为空!")
        #     return HttpResponseRedirect(reverse('SNA:login'))

        # 验证账号密码正确
        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            return HttpResponseRedirect(reverse('SNA:index'))
        else:
            messages.success(request, "密码错误")
            return HttpResponseRedirect(reverse('SNA:login'))


def index(req):
    """默认转发页面，展示最多20个最新的数据集
    点击数据集名可以跳转到数据集详情页"""
    if not req.user.is_authenticated:
        return HttpResponseRedirect(reverse('SNA:login'))

    lasted_pub_lt = Dataset.objects.order_by("pub_date")
    # 仅显示当前登录用户拥有的数据集，及公开数据集
    lasted_pub_lt = [e for e in lasted_pub_lt if e.is_private is False
                     or (e.owner and e.owner.pk == req.user.pk)]

    template = loader.get_template("SNA/index.html")

    # 首页显示当前用户的一些信息
    user_info = {
        "last_login": req.user.last_login + timedelta(hours=8),
        "num_own_dataset": len(Dataset.objects.filter(owner=req.user)),
        "num_run_record": len(RunResult.objects.filter(exec_er=req.user)),
    }

    # 关联HTML的内容, the context is a dictionary
    # mapping template variable names to Python objects.
    context = {
        "dataset_lt": lasted_pub_lt,
        "cur_username": req.user.username,
        "user_info": user_info,
        "sys_meta": SYS_META,
    }
    return HttpResponse(
        template.render(context, req))


def dataset_detail(req, dataset_id):
    """数据集详情页的视图，
    如果数据集id对应的数据集不存在，则转发到404异常"""
    if not req.user.is_authenticated:
        return HttpResponseRedirect(reverse('SNA:login'))

    try:
        dataset = Dataset.objects.get(pk=dataset_id)
        run_records = RunResult.objects.filter(dataset_used=dataset)

    except Dataset.DoesNotExist:
        raise Http404("Dataset does not exist.")
    except RunResult.DoesNotExist:
        run_records = []

    try:
        d_size_str = get_file_size(f"./SNA/dataset_store/{dataset.pk}.{dataset.path.split('.')[-1]}")
    except Exception as e:
        print(e)
        d_size_str = "未知"

    # 获取算法运行状态
    for record in run_records:
        try:
            if record.alg_name == "TILES":
                s = open(f"./SNA/alg_result/{record.pk}/tiles.tag").read()
            elif record.alg_name == "TGAT":
                s = ""
                pass
        except FileNotFoundError:
            s = "未知"
        # run_status_map[record.pk] = s
        # 动态绑定
        record.status = s
        record.show_result = s == "已完成"

    owner = dataset.owner.username if dataset.owner else "未知"

    # 调用template更快捷的方法：使用render
    return render(req, "SNA/detail.html",
                  {"dataset": dataset,
                   "d_size_str": d_size_str,
                   "run_records": run_records,
                   "owner": owner,
                   "cur_username": req.user.username,
                   "access_str": "私有" if dataset.is_private else "公开",
                   "own": dataset.owner and dataset.owner.pk == req.user.pk,
                   "sys_meta": SYS_META})


def dataset_upload(req):
    # 此处应当处理文件传输和存储
    # 此处使用redirect（重定向）而不是response是为了防止用户点击"返回"导致数据被提交两次
    if not req.user.is_authenticated:
        return HttpResponseRedirect(reverse('SNA:login'))

    BASE_DIR = "SNA/dataset_store/"
    obj = req.FILES.get('filename', '1')

    model = Dataset(name=req.POST.get('dataset_name'), path=obj.name,
                    pub_date=datetime.now() + timedelta(hours=8), owner=req.user,
                    is_private=req.POST.get('is_private') is None)
    model.save()

    # 改名，防止文件重名
    f = open(os.path.join(BASE_DIR, str(model.pk) + "." + obj.name.split(".")[-1]), 'wb')

    for chunk in obj.chunks():
        f.write(chunk)
    f.close()

    return HttpResponseRedirect(
        reverse('SNA:index'))


def dataset_alter(req):
    """修改一个已有数据集的某些字段"""
    if not req.user.is_authenticated:
        return HttpResponseRedirect(reverse('SNA:login'))
    dataset_id = req.POST.get("dataset_id")
    dataset_model = get_object_or_404(Dataset, pk=dataset_id)

    # 修改模型对象的"访问权限"字段
    is_private = req.POST.get("access-switch") is None
    dataset_model.is_private = is_private
    # 提交到数据库
    dataset_model.save()

    # 使用重定向防止表单重复提交
    return HttpResponseRedirect(
        reverse('SNA:d_detail', args=(dataset_id,)))


def run_alg(req):
    if not req.user.is_authenticated:
        return HttpResponseRedirect(reverse('SNA:login'))

    task_name = req.POST.get("task_select")
    dataset_id = req.POST.get("dataset_id")
    dataset = get_object_or_404(Dataset, pk=dataset_id)
    alg_name_map = {"社区演化分析": "TILES", "节点联系预测": "TGAT"}
    alg_name = alg_name_map[task_name]

    record_model = RunResult(exec_datetime=datetime.now() + timedelta(hours=8),
                             dataset_used=dataset,
                             alg_name=alg_name, task_name=task_name,
                             exec_er=req.user)
    record_model.save()

    # 开启子进程执行算法
    if alg_name == "TILES":
        dump_dir = "./SNA/alg_result/" + str(record_model.pk) + "/"
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        p = Process(target=run_tiles,
                    args=(BASE_DIR + str(dataset.pk) + "." + dataset.path.split(".")[-1],
                          dump_dir))
    elif alg_name == "TGAT":
        pass
    p.start()

    # 刷新详情页
    return HttpResponseRedirect(
        reverse('SNA:d_detail', args=(dataset.id,)))


def tiles_plot(req):
    """TILES的结果的可视化"""
    # 数据构造
    record_id = req.POST.get("record_id")
    record = get_object_or_404(RunResult, pk=record_id)
    data = get_network_meta(f"SNA/alg_result/{record_id}/")
    linked_dataset = RunResult.objects.get(pk=record_id).dataset_used

    try:
        d_size_str = get_file_size(f"./SNA/dataset_store/{linked_dataset.pk}.{linked_dataset.path.split('.')[-1]}")
    except Exception as e:
        print(e)
        d_size_str = "未知"

    # 结果解析数据
    comm_lt = get_some_comm_id(f"SNA/alg_result/{record_id}/")

    # 转发到可视化页面的模板
    return render(req, "SNA/tiles_plot.html",
                  {
                      "data": data,
                      "sys_meta": SYS_META,
                      "dataset_id": linked_dataset.pk,
                      "cur_username": req.user.username,
                      "dataset": linked_dataset,
                      "d_size_str": d_size_str,
                      "record": record,
                      "comm_lt": comm_lt,
                  })


def about_sys(req):
    """关于系统"""
    return render(req, "SNA/about.html", {
        "cur_username": req.user.username,
        "sys_meta": SYS_META,
    })


def dataset_lt_single(req):
    # 分开自己创建的数据集和公开数据集
    own_lt = Dataset.objects.filter(owner=req.user)
    pub_lt = Dataset.objects.filter(is_private=False)
    return render(req, "SNA/dataset_lt.html",
                  {
                      "sys_meta": SYS_META,
                      "own_lt": own_lt,
                      "pub_lt": pub_lt,
                      "cur_username": req.user.username,
                  })


def dataset_upload_single(req):
    return render(req, "SNA/dataset_upload.html",
                  {
                      "sys_meta": SYS_META,
                      "cur_username": req.user.username,
                  })


def comm_detail(req, r_id, c_id, s_id):
    record = get_object_or_404(RunResult, pk=r_id)
    comm_data = {
        "c_id": c_id,
    }

    # 设置暂存目录，提高访问速度
    if f"{r_id}_{c_id}.json" not in os.listdir("SNA/comm_json_dump/"):
        slices = get_slices_and_data(f"SNA/alg_result/{r_id}/", c_id)
        json.dump(slices, open(f"SNA/comm_json_dump/{r_id}_{c_id}.json", "w", encoding="utf8"), indent=2)
    else:
        slices = json.load(open(f"SNA/comm_json_dump/{r_id}_{c_id}.json", "r", encoding="utf8"))

    # sid=0表示社区的第一个时间片 -1表示社区最后一个时间片
    first_s = slices[0]["slice_id"]
    last_s = slices[-1]["slice_id"]
    if s_id == 0 or s_id == "0":
        s_id = first_s
    elif s_id == -1 or s_id == "-1":
        s_id = last_s

    return render(req, "SNA/comm_detail.html",
                  {
                      "sys_meta": SYS_META,
                      "cur_username": req.user.username,
                      "comm_data": comm_data,
                      "record": record,
                      "slices": slices,
                      "s_id": s_id,
                      "pre_s_id": None if s_id == first_s else s_id - 1,
                      "next_s_id": None if s_id == last_s else s_id + 1,
                  })


def get_comm_json_data(req, r_id, c_id, s_id):
    slices = json.load(open(f"SNA/comm_json_dump/{r_id}_{c_id}.json", "r", encoding="utf8"))
    start_s = slices[0]["slice_id"]
    cur_slice = slices[s_id - start_s]

    nodes = [{"name": m, "value": 1, "categories": 0} for m in cur_slice["members"]]
    id_map = {m: i for i, m in enumerate(cur_slice["members"])}
    links = [{"source": id_map[edge[0]], "target": id_map[edge[1]]} for edge in cur_slice["edgelist"]]

    json_data = {
        "type": "force",
        "categories": [
            {
                "name": "Member",
                "keyword": {},
                "base": "Member"
            },
        ],
        "nodes": nodes,
        "links": links,
    }
    return JsonResponse(json_data)
