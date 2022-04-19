import os

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect
from .models import Dataset, RunResult
from django.template import loader
from django.urls import reverse
from datetime import datetime, timedelta
from multiprocessing import Process
from SNA.algorithms.run_tiles import run_tiles


# Create your views here.
# view要么返回HttpResponse，要么返回HttpException

# view可以通过generic的方式来继承


def index(req):
    """默认转发页面，展示最多5个最新的数据集
    点击数据集名可以跳转到数据集详情页"""
    # return HttpResponse("hello!")
    lasted_pub_lt = Dataset.objects.order_by("pub_date")[:20]
    template = loader.get_template("SNA/index.html")
    # 关联HTML的内容, the context is a dictionary
    # mapping template variable names to Python objects.
    context = {
        "dataset_lt": lasted_pub_lt
    }
    return HttpResponse(
        template.render(context, req))


def dataset_detail(req, dataset_id):
    """数据集详情页的视图，
    如果数据集id对应的数据集不存在，则转发到404异常"""
    try:
        dataset = Dataset.objects.get(pk=dataset_id)
        run_records = RunResult.objects.filter(dataset_used=dataset)

    except Dataset.DoesNotExist:
        raise Http404("Dataset does not exist.")
    except RunResult.DoesNotExist:
        run_records = []

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
        record.status = s

    # 调用template更快捷的方法：使用render
    return render(req, "SNA/detail.html",
                  {"dataset": dataset,
                   "run_records": run_records})


def dataset_upload(req):
    # 此处应当处理文件传输和存储
    # 此处使用redirect（重定向）而不是response是为了防止用户点击"返回"导致数据被提交两次
    # return HttpResponseRedirect(
    #     reverse('SNA:d_detail', args=(dataset.id,)))

    BASE_DIR = "SNA/dataset_store/"
    obj = req.FILES.get('filename', '1')
    print(obj.name)

    model = Dataset(name=req.POST.get('dataset_name'), path=os.path.join(BASE_DIR, obj.name),
                    pub_date=datetime.now() + timedelta(hours=8))  # todo::添加用户信息

    f = open(os.path.join(BASE_DIR, obj.name), 'wb')

    for chunk in obj.chunks():
        f.write(chunk)
    f.close()
    model.save()

    return HttpResponseRedirect(
        reverse('SNA:index'))


def run_alg(req):
    task_name = req.POST.get("task_select")
    dataset_id = req.POST.get("dataset_id")
    dataset = get_object_or_404(Dataset, pk=dataset_id)
    alg_name_map = {"社区演化分析": "TILES", "节点联系预测": "TGAT"}
    alg_name = alg_name_map[task_name]

    record_model = RunResult(exec_datetime=datetime.now() + timedelta(hours=8),
                             dataset_used=dataset,
                             alg_name=alg_name, task_name=task_name)
    record_model.save()

    # 开启子进程执行算法
    if alg_name == "TILES":
        dump_dir = "./SNA/alg_result/" + str(record_model.pk) + "/"
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        p = Process(target=run_tiles, args=(dataset.path, dump_dir))
    elif alg_name == "TGAT":
        pass
    p.start()

    # 刷新详情页
    return HttpResponseRedirect(
        reverse('SNA:d_detail', args=(dataset.id,)))


# def run_tiles(req, dataset_id):
#     dataset = get_object_or_404(Dataset, pk=dataset_id)
#
#     # 运行TILES算法
#     obs = 30
#     ttl = 240
#     filename = ""  # todo::设置用户上传文件的存放路径
#     dump_dir = ""  # todo::设置算法执行结果的存放路径
#     model = TILES(obs=obs, ttl=ttl, filename=filename, path=dump_dir)
#     model.execute()
#
#     # 返回数据集详情页
#     return HttpResponseRedirect(
#         reverse('SNA:d_detail', args=(dataset.id,)))
#
#
# def run_tgat(req, dataset_id):
#     dataset = get_object_or_404(Dataset, pk=dataset_id)
#
#     # todo::调用命令行执行tgat
#     pass
#
#     # 返回数据集详情页
#     return HttpResponseRedirect(
#         reverse('SNA:d_detail', args=(dataset.id,)))
