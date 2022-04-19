# -*- coding: utf-8 -*-
# @Time: 2020/12/13 15:37
# @Author: Rollbear
# @Filename: node_attributed_TILES.py
import csv
import datetime
import gzip
import os
import time
from io import StringIO
from queue import PriorityQueue

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from future.utils import iteritems
from tqdm import tqdm

from SNA.algorithms.tiles.util.timer import timer

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"

DEBUG = True


class NodeAttrBase:
    """handler基类"""

    def get_distance(self, node_a, node_b):
        pass


class NodeAttrHandler(NodeAttrBase):
    def __init__(self, node_attr_filename, alpha, beta, segregate_char=","):
        """
        :param node_attr_filename: 节点属性表路径
        :param alpha: α
        :param beta: β
        :param segregate_char: 属性分隔符
        """
        self.node_attr = node_attr_filename
        self.alpha = alpha
        self.beta = beta
        self.distance_matrix = {}
        self.segregate_char = segregate_char

        self.build_distance_matrix()

    def build_distance_matrix(self):
        with open(self.node_attr, "r") as rf:
            for node_id, node_attr in enumerate(rf):
                # node_id从1开始编号
                self.distance_matrix[node_id + 1] = \
                    np.array([float(elem) for elem in str(node_attr).rstrip().split(self.segregate_char)])

    def get_distance(self, node_a, node_b):
        """基于属性的节点距离计算，目前使用欧氏距离"""
        return np.sqrt(np.sum((self.distance_matrix[node_a] - self.distance_matrix[node_b]) ** 2))

    def init_weight(self, node_a, node_b):
        return self.alpha + self.beta * self.get_distance(node_a, node_b)


class DynamicNodeAttrHandler(NodeAttrBase):
    def __init__(self, stream_source, attr_tag: str, segregate="\t"):
        # todo::从这里加载数据，而不是TILES对象中
        self._node_dict = {}
        self._attr_tag = attr_tag
        self._segregate = segregate
        self._source_type = None

        if isinstance(stream_source, str):
            self._source_type = "file"
            self._source = open(stream_source, "r", encoding="utf8")
        else:
            self._source_type = "extern_gen"
            self._source = iter(stream_source)

    def update_node_attr(self, node_id, value, mode="avg"):
        if mode == "avg":
            if node_id in self._node_dict:
                self._node_dict[node_id] = (self._node_dict[node_id] + value) / 2
            else:
                self._node_dict[node_id] = value

    def get_distance(self, node_a, node_b):
        if node_a not in self._node_dict or node_b not in self._node_dict:
            return 0
        # 返回欧氏距离的倒数（距离越近得分越高）
        dist = 1 / np.sqrt(np.sum((self._node_dict[node_a] - self._node_dict[node_b]) ** 2))
        return dist

    def __next__(self):
        line = self._source.__next__()
        if line.startswith(self._attr_tag):
            # 节点属性增量流，更新节点字典
            line_tuple = line.split(self._segregate)
            node_id, vec_str = line_tuple[1], line_tuple[2]
            vec = np.asarray(vec_str.split(), dtype='float32')
            self.update_node_attr(node_id=node_id, value=vec)
            return self.__next__()  # 检查下一行
        else:
            return line  # 边流直接返回

    def __iter__(self):
        try:
            while True:
                yield self.__next__()
        except StopIteration:
            pass


class TwitterCSVDataHandler:
    """从多个csv文件中构造TILES所需的流数据，数据由NodeAttrHandler接收"""

    def __init__(self, csv_dir: str, attr_line_tag: str, segregate: str,
                 vec_late_load=True, pool_max=None, low_memory_mode=False):
        """
        1. 将所有csv中的推特行混合后按照时间从早到晚排序
        2. 生成流
        """
        self._csv_dir = csv_dir if csv_dir.endswith("/") else csv_dir + "/"
        self._attr_line_tag = attr_line_tag
        self._segregate = segregate
        self._vec_late_load = vec_late_load
        self._pool_max = pool_max
        self._low_memory_mode = low_memory_mode

        self._data = self._integration_and_sort()

        # 新增：使用文件池来提高读取效率
        if self._vec_late_load:
            self._pool = self.FilePool(max_num_file=self._pool_max, files_dir=self._csv_dir)

    class FilePool:
        def __init__(self, max_num_file: int, files_dir: str):
            self.queue = []
            self.files = {}
            self.max_num_file = max_num_file
            self.files_dir = files_dir if files_dir.endswith("/") else self.files_dir + "/"
            self.hit_count = {}

            print("loading file pool...")
            self._load_to_max()  # 初始化文件池

        def _load_to_max(self):
            for filename in os.listdir(self.files_dir):
                path = self.files_dir + filename
                if path not in self.files:
                    with open(path, "r", encoding="utf8") as rf:
                        self.files[path] = list(csv.reader(rf))[1:]  # 跳过表头
                        self.queue.append(path)
                    if len(self.files) >= self.max_num_file:
                        break

        def __getitem__(self, item: str):
            if item not in self.files:
                # 若缓存未命中
                # 将缓存中的文件按照已命中的次数升序排列，然后释放队头（已命中次数最少的文件）
                # 已测试FIFO换出策略和最多命中换出策略，效率均没有最少命中换出策略高
                self.queue.sort(reverse=False, key=lambda elem: self.hit_count.get(elem, 0))
                self.files.pop(self.queue[0])
                self.queue.pop(0)

                # 从磁盘加载未命中的文件
                print(f"{item} no hit! loading from disk... (cur pool size: {len(self.files)})")
                with open(item, "r", encoding="utf8") as rf:
                    self.files[item] = list(csv.reader(rf))[1:]  # 跳过表头
                    self.queue.append(item)

            # 如果缓存命中，该文件计数加一（磁盘上的文件换入后命中也算命中）
            self.hit_count[item] = self.hit_count.get(item, 0) + 1
            return self.files[item]

    def dump_stream_file(self, dump_path):
        with open(dump_path, "w", encoding="utf8") as wf:
            for line in iter(self):
                wf.write(line)

    def __iter__(self):
        if self._vec_late_load:
            # 若使用低内存模式，mentions列表为后加载
            if self._low_memory_mode:
                for tweet in tqdm(self._data, desc="stream: "):
                    # 构造节点属性流：节点id和表示向量
                    # 从文件池中按先前生成的索引取回一行
                    row = self._pool[tweet[1][0]][tweet[1][1]]
                    # 则此处每条tweet的表示向量从之前生成的文件索引中读取
                    node_attr_line = self._attr_line_tag + self._segregate + row[1] \
                                     + self._segregate + row[-1]

                    # 构造边流：起点、终点、时间戳
                    elt_lines = []
                    # 构造边终点
                    tmp = row[14][2:-2]
                    if "', '" in tmp:
                        mentions = list(tmp.split("', '"))
                    elif tmp == "":
                        mentions = []
                    else:
                        mentions = [tmp]
                    in_reply_to_user = row[5]
                    quoted_user = row[8]
                    retweeted_user = row[11]
                    for uid in [in_reply_to_user, quoted_user, retweeted_user]:
                        if uid not in mentions and uid != "None":
                            mentions.append(uid)
                    for dest in mentions:  # 更新后的mentions列表
                        elt_lines.append(row[1] + self._segregate + dest + self._segregate + str(tweet[-1]))

                    # 写入文本文件，先写入节点属性增量，后写入边流
                    yield node_attr_line + "\n"
                    for e in elt_lines:
                        yield e + "\n"
            # 不使用低内存模式，则mentions在文件聚合时已经计算
            else:
                for tweet in tqdm(self._data, desc="stream: "):
                    # 构造节点属性流：节点id和表示向量
                    # 则此处每条tweet的表示向量从之前生成的文件索引中读取
                    node_attr_line = self._attr_line_tag + self._segregate + tweet[0] \
                                     + self._segregate + self._pool[tweet[1][0]][tweet[1][1]][-1]
                    # 构造边流：起点、终点、时间戳
                    elt_lines = []
                    for dest in tweet[2]:  # 更新后的mentions列表
                        elt_lines.append(tweet[0] + self._segregate + dest + self._segregate + str(tweet[-1]))

                    # 写入文本文件，先写入节点属性增量，后写入边流
                    yield node_attr_line + "\n"
                    for e in elt_lines:
                        yield e + "\n"

        else:
            for tweet in tqdm(self._data, desc="stream: "):
                # 构造节点属性流：节点id和表示向量
                node_attr_line = self._attr_line_tag + self._segregate + tweet[0] \
                                 + self._segregate + tweet[1]
                # 构造边流：起点、终点、时间戳
                elt_lines = []
                for dest in tweet[2]:  # 更新后的mentions列表
                    elt_lines.append(tweet[0] + self._segregate + dest + self._segregate + str(tweet[-1]))

                # 写入文本文件，先写入节点属性增量，后写入边流
                yield node_attr_line + "\n"
                for e in elt_lines:
                    yield e + "\n"

    def _integration_and_sort(self):
        data_to_sort = []
        for csv_path in tqdm(os.listdir(self._csv_dir), desc="loading and integrating csv: "):
            if csv_path.endswith(".csv"):
                is_header = True
                with open(self._csv_dir + csv_path, "r", encoding="utf8") as rf:
                    reader = csv.reader(rf)
                    for row_id, row in enumerate(reader):
                        if is_header:
                            is_header = False
                            continue  # 跳过首行（表头）

                        vec = row[-1]  # 推特文本向量
                        # 若设置vec_late_load，排序时不保存推特文本向量，仅保存一个索引，排序完成后再加载推特向量
                        user_id = row[1]  # 边起点

                        vec_file_path = self._csv_dir + csv_path
                        vec_line_id = row_id - 1
                        vec_index = (vec_file_path, vec_line_id)

                        # 若低内存模式未启用，构造边终点；否则mentions为后加载的形式
                        if not self._low_memory_mode:
                            tmp = row[14][2:-2]
                            if "', '" in tmp:
                                mentions = list(tmp.split("', '"))
                            elif tmp == "":
                                mentions = []
                            else:
                                mentions = [tmp]
                            in_reply_to_user = row[5]
                            quoted_user = row[8]
                            retweeted_user = row[11]
                            for uid in [in_reply_to_user, quoted_user, retweeted_user]:
                                if uid not in mentions and uid != "None":
                                    mentions.append(uid)

                        # 构造时间戳
                        datetime_str = row[4]  # Wed Apr 10 12:00:34 +0000 2019
                        datetime_elem = datetime_str.split(" ")
                        month_map = {
                            "Jan": "1",
                            "Feb": "2",
                            "Mar": "3",
                            "Apr": "4",
                            "May": "5",
                            "Jun": "6",
                            "Jul": "7",
                            "Aug": "8",
                            "Sep": "9",
                            "Oct": "10",
                            "Nov": "11",
                            "Dec": "12"
                        }
                        time_array = time.strptime(
                            f"{datetime_elem[-1]}-{month_map[datetime_elem[1]]}-{datetime_elem[2]} {datetime_elem[3]}",
                            "%Y-%m-%d %H:%M:%S")
                        time_stamp = time.mktime(time_array)

                        if self._vec_late_load:
                            if self._low_memory_mode:
                                data_to_sort.append([None, vec_index, None, time_stamp])
                            else:
                                data_to_sort.append([user_id, vec_index, mentions, time_stamp])
                        else:
                            data_to_sort.append([user_id, vec, mentions, time_stamp])

        print("sorting tweet prefix...")
        data_to_sort.sort(key=lambda elem: elem[-1])  # 按照时间戳排序
        return data_to_sort


class NodeAttributedTILES(object):

    def __init__(self, filename=None, g=None, ttl=float('inf'), obs=7,
                 path="", start=None, end=None,
                 node_attr_handler: NodeAttrBase = None, auto_timestamp=False, segregate_char="\t",
                 remove_threshold=1, remove_step=1, plot_node_simi=False, count_avg_cc=False):
        """
            Constructor
            :param g: networkx graph
            :param ttl: edge time to live (days)
            :param obs: observation window (days)
            :param path: Path where generate the results and find the edge file
            :param start: starting date
            :param end: ending date
            :param node_attr_handler: 节点属性处理对象
            :param auto_timestamp: 是否将静态图转化为动态图（自动附加时间戳）
            :param segregate_char: 表分隔符，默认为制表符
        """
        self.path = path
        self.ttl = ttl
        self.cid = 0
        self.actual_slice = 0
        if g is None:
            self.g = nx.Graph()
        else:
            self.g = g
        self.splits = None
        self.spl = StringIO()
        self.base = os.getcwd()
        self.status = open("%s/%s/extraction_status.txt" % (self.base, path), "w")
        self.removed = 0
        self.added = 0
        self.filename = filename
        self.start = start
        self.end = end
        self.obs = obs
        self.count_avg_cc = count_avg_cc
        self.communities = {}

        # 新增的参数
        self.auto_timestamp = auto_timestamp
        self.segregate_char = segregate_char
        self.node_attr_handler = node_attr_handler
        self.remove_threshold = remove_threshold
        self.remove_step = remove_step
        self.plot_node_simi = plot_node_simi

        # 新增的字段
        self.slices_cc = []

    def execute(self):
        """
            Execute TILES algorithm
            流程顺序：
            0. 初始化（进入主循环之前）：初始化边移除队列、最初时间戳
            主循环（直到stream source到达尽头）：
            1. 以新边时间戳作为本次迭代的最新时间
            2. 新边加入移除队列，但先不加入图
            3. 基于本次迭代的最新时间判断是否开启一个观察点，即输出缓存中的数据到文件
            4. 检查移除队列，移除过期的边，并更新社区关系
            5. 新边加入图（在数据结构中注册节点、注册边、更新权值）
            6. 计算新边加入后的社区关系（核心点传播/新社区/合并）
        """
        self.status.write(u"Started! (%s) \n\n" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()

        qr = PriorityQueue()  # 用于存放等待消亡的边的优先级队列
        cur_timestamp = time.time()  # 用于自动附加时间戳
        count = 0

        #################################################
        #                   Main Cycle                  #
        #################################################

        last_break = actual_time = None

        # 进入主循环，每一次迭代即为处理一条边
        is_first_line = True
        if isinstance(self.node_attr_handler, DynamicNodeAttrHandler):
            stream_source = self.node_attr_handler
        else:
            stream_source = open(self.filename)
        for line in stream_source:

            if is_first_line:
                # 第一行需要初始化的变量
                if self.auto_timestamp:
                    first_line = line + self.segregate_char + str(cur_timestamp)
                    cur_timestamp += 86400  # 自动附加时间戳时，每条边插入的时间相差一天
                else:
                    first_line = line

                # 从时间戳获得datetime
                # 负数时间戳分开处理
                if float(first_line.split(self.segregate_char)[2]) > 0:
                    actual_time = datetime.datetime.fromtimestamp(float(first_line.split(self.segregate_char)[2]))
                else:
                    actual_time = datetime.datetime(1970, 1, 1) + \
                                  datetime.timedelta(seconds=float(first_line.split(self.segregate_char)[2]))

                last_break = actual_time
                is_first_line = False

            if self.auto_timestamp:
                line = line.split(self.segregate_char) + [str(cur_timestamp)]
                cur_timestamp += 86400  # 自动附加时间戳时，每条边插入的时间相差一天
            else:
                line = line.split(self.segregate_char)
            self.added += 1
            e = {}
            u = int(line[0])
            v = int(line[1])
            # 当前边的datetime时间，负数时间戳分开处理
            if float(line[2]) > 0:
                dt = datetime.datetime.fromtimestamp(float(line[2]))
            else:
                dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(line[2]))

            e["u"] = u
            e["v"] = v
            if self.node_attr_handler:
                if isinstance(self.node_attr_handler, NodeAttrHandler):
                    e['weight'] = self.node_attr_handler.init_weight(node_a=e["u"], node_b=e["v"])
                elif isinstance(self.node_attr_handler, DynamicNodeAttrHandler):
                    e["weight"] = stream_source.get_distance(node_a=e["u"], node_b=e["v"])

            else:
                e['weight'] = 1
            # month = dt.month

            #############################################
            #               Observations                #
            #############################################

            gap = dt - last_break
            dif = gap.days

            # 如果当前边的时间距离上一条边的时间已经大于obs（天为单位），
            # 则开启一个观察点
            if dif >= self.obs:
                last_break = dt
                self.added -= 1

                print("New slice. Starting Day: %s" % dt)

                self.status.write(u"Saving Slice %s: Starting %s ending %s - (%s)\n" %
                                  (self.actual_slice, actual_time, dt,
                                   str(time.asctime(time.localtime(time.time())))))

                self.status.write(u"Edge Added: %d\tEdge removed: %d\n" % (self.added, self.removed))
                self.added = 1
                self.removed = 0

                actual_time = dt
                self.status.flush()

                self.splits = gzip.open("%s/%s/splitting-%d.gz" % (self.base, self.path, self.actual_slice), "wt", 3)
                self.splits.write(self.spl.getvalue())
                self.splits.flush()
                self.splits.close()
                self.spl = StringIO()

                self.print_communities()  # 输出当前观察窗口的数据到文件
                self.status.write(
                    u"\nStarted Slice %s (%s)\n" % (self.actual_slice, str(datetime.datetime.now().time())))

            if u == v:
                continue

            # Check if edge removal is required
            if self.ttl != float('inf'):
                # 新边加入优先级队列
                qr.put((dt, (int(e['u']), int(e['v']), float(e['weight']))))
                self.remove(dt, qr)  # 移除当前时间过期的边

            # 对两个端点u与v：如果是新出现的端点，则在数据结构中注册
            if not self.g.has_node(u):
                self.g.add_node(u)
                self.g.node[u]['c_coms'] = {}  # central
            if not self.g.has_node(v):
                self.g.add_node(v)
                self.g.node[v]['c_coms'] = {}

            # 如果当前边已经在图中存在，则它的权值更新，否则在数据结构中注册这条边
            if self.g.has_edge(u, v):
                w = self.g.adj[u][v]["weight"]
                self.g.adj[u][v]["weight"] = w + e['weight']
                continue
            else:
                self.g.add_edge(u, v)
                self.g.adj[u][v]["weight"] = e['weight']

            # 获得当前边两端点的邻接点
            u_neighbours = list(self.g.neighbors(u))
            v_neighbours = list(self.g.neighbors(v))

            #############################################
            #               Evolution                   #
            #############################################

            # new community of peripheral nodes (new nodes)
            if len(u_neighbours) > 1 and len(v_neighbours) > 1:
                common_neighbors = set(u_neighbours) & set(v_neighbours)
                self.common_neighbors_analysis(u, v, common_neighbors)  # 若u与v没有共同邻居，该函数不执行任何操作

            count += 1

        #  Last writing
        # 算法终止，写入最后一个观测点
        self.status.write(u"Slice %s: Starting %s ending %s - (%s)\n" %
                          (self.actual_slice, actual_time, actual_time,
                           str(time.asctime(time.localtime(time.time())))))
        self.status.write(u"Edge Added: %d\tEdge removed: %d\n" % (self.added, self.removed))
        self.added = 0
        self.removed = 0

        self.print_communities()
        self.status.write(u"Finished! (%s)" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()
        self.status.close()

    @property
    def new_community_id(self):
        """
            Return a new community identifier
            申请一个新的社区id，唯一标识每一个社区
            :return: new community id
        """
        # 上一个发放的社区ID自增作为新的社区ID
        self.cid += 1
        # 在社区表中注册新的社区ID，节点表初始化为空
        self.communities[self.cid] = {}
        # 返回新的社区ID
        return self.cid

    def remove(self, actual_time, qr):
        """
            Edge removal procedure
            边移除模块，从优先级队列中移除过期的边
            :param actual_time: timestamp of the last inserted edge
            :param qr: Priority Queue containing the edges to be removed ordered by their timestamps
        """

        coms_to_change = {}
        at = actual_time

        # main cycle on the removal queue
        if not qr.empty():

            t = qr.get()  # 元素出队列（移除）
            timestamp = t[0]
            e = (t[1][0], t[1][1], t[1][2])

            delta = at - timestamp
            displacement = delta.days

            if displacement < self.ttl:
                qr.put((timestamp, t[1]))

            else:
                while self.ttl <= displacement:

                    self.removed += 1
                    u = int(e[0])
                    v = int(e[1])
                    if self.g.has_edge(u, v):

                        w = self.g.adj[u][v]["weight"]

                        # decreasing link weight if greater than one
                        # (multiple occurrence of the edge: remove only the oldest)
                        # 如果过期的边权值大于1：使它的权减1，并刷新时间戳
                        if w > self.remove_threshold:
                            self.g.adj[u][v]["weight"] = w - self.remove_step
                            e = (u, v, w - 1)
                            qr.put((at, e))

                        # 否则（即边在减之间权值为1），移除这条边，并执行后续的社区更新
                        else:
                            # u and v shared communities
                            if len(list(self.g.neighbors(u))) > 1 and len(list(self.g.neighbors(v))) > 1:
                                # coms: 两端点都共同归属的社区的id
                                coms = set(self.g.node[u]['c_coms'].keys()) & set(self.g.node[v]['c_coms'].keys())

                                for c in coms:
                                    # 计算coms_to_change：被移除的边涉及的点集
                                    if c not in coms_to_change:
                                        # 两端点的共同邻居节点
                                        common_neighbours = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                                        coms_to_change[c] = [u, v]
                                        coms_to_change[c].extend(list(common_neighbours))
                                    else:
                                        common_neighbours = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                                        coms_to_change[c].extend(list(common_neighbours))
                                        coms_to_change[c].extend([u, v])
                                        ctc = set(coms_to_change[c])  # 去重
                                        coms_to_change[c] = list(ctc)
                            else:
                                # 若u除了v没有其他邻居，则从现在的社区中移除u（因为这条边消失后u成为游离点）
                                if len(list(self.g.neighbors(u))) < 2:
                                    coms_u = [x for x in self.g.node[u]['c_coms'].keys()]
                                    for cid in coms_u:
                                        self.remove_from_community(u, cid)
                                # 若v除了u没有其他邻居，同样处理
                                if len(list(self.g.neighbors(v))) < 2:
                                    coms_v = [x for x in self.g.node[v]['c_coms'].keys()]
                                    for cid in coms_v:
                                        self.remove_from_community(v, cid)

                            self.g.remove_edge(u, v)

                    if not qr.empty():
                        t = qr.get()

                        timestamp = t[0]
                        delta = at - timestamp
                        displacement = delta.days

                        e = t[1]

        # update of shared communities
        # 边移除后，更新它涉及到的点集（字典{社区id: 该社区中需要更新的点}）
        self.update_shared_coms(coms_to_change)

    def update_shared_coms(self, coms_to_change):
        """
        update of shared communities
        有边消失后，更新它涉及的社区
        :param coms_to_change: 该边涉及的社区
        :return: None
        """
        for c in coms_to_change:
            if c not in self.communities:
                continue

            c_nodes = self.communities[c].keys()

            if len(c_nodes) > 3:

                sub_c = self.g.subgraph(c_nodes)
                # 调用nx的api来计算子图中的连通部分的数量
                c_components = nx.number_connected_components(sub_c)

                # unbroken community
                # 社区不分裂时（社区中的点仍然是一个连通整体则肯定没有分裂）
                if c_components == 1:
                    # 仅更新涉及的社区中涉及的点即可
                    to_mod = sub_c.subgraph(coms_to_change[c])
                    self.modify_after_removal(to_mod, c)

                # broken community: bigger one maintains the id, the others obtain a new one
                # 社区分裂时：最大的子社区获得原来的社区id，其他子社区申请新id
                # 仅当社区完全断裂，成为不连通的两部分才认为分裂
                else:
                    new_ids = []

                    first = True
                    components = nx.connected_components(sub_c)
                    for com in components:
                        if first:
                            if len(com) < 3:
                                self.destroy_community(c)
                            else:
                                to_mod = list(set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)
                                self.modify_after_removal(sub_c, c)
                            first = False

                        else:
                            if len(com) > 3:
                                # update the memberships: remove the old ones and add the new one
                                to_mod = list(set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)

                                central = self.centrality_test(sub_c).keys()
                                if len(central) >= 3:
                                    # 申请新的社区id
                                    actual_id = self.new_community_id
                                    # 新申请的id的集合，稍后写入split记录文件中
                                    # （所以split记录中每行方括号后的id都是在这一步新申请的id）
                                    new_ids.append(actual_id)
                                    for n in central:
                                        self.add_to_community(n, actual_id)

                    # splits
                    if len(new_ids) > 0 and self.actual_slice > 0:
                        self.spl.write(u"%s\t%s\n" % (c, str(new_ids)))
            else:
                self.destroy_community(c)

    def modify_after_removal(self, sub_c, c):
        """
            Maintain the clustering coefficient invariant after the edge removal phase
            某条边移除后的更新机制，更新社区成员身份
            :param sub_c: sub-community to evaluate
            :param c: community id
        """
        central = self.centrality_test(sub_c).keys()

        # in case of previous splits, update for the actual nodes
        remove_node = set(self.communities[c].keys()) - set(sub_c.nodes())

        for rm in remove_node:
            self.remove_from_community(rm, c)

        # 核心点>=3才能组成一个社区
        if len(central) < 3:
            self.destroy_community(c)
        else:
            not_central = set(sub_c.nodes()) - set(central)
            for n in not_central:
                # 将不满足核心点要求的点从核心点中移除
                self.remove_from_community(n, c)

    def common_neighbors_analysis(self, u, v, common_neighbors):
        """
            General case in which both the nodes are already present in the net.
            处理新边加入后的社区更新（标签传播）
            :param u: a node
            :param v: a node
            :param common_neighbors: common neighbors of the two nodes，两端点的共同邻居节点
        """

        # no shared neighbors
        if len(common_neighbors) < 1:
            return

        else:
            # shared_coms: 端点u与v共同归属的社区
            shared_coms = set(self.g.node[v]['c_coms'].keys()) & set(self.g.node[u]['c_coms'].keys())
            # only: 仅由某个端点归属而另一个端点不归属的社区
            only_u = set(self.g.node[u]['c_coms'].keys()) - set(self.g.node[v]['c_coms'].keys())
            only_v = set(self.g.node[v]['c_coms'].keys()) - set(self.g.node[u]['c_coms'].keys())

            # community propagation: a community is propagated iff at least two of [u, v, z] are central
            propagated = False  # 该布尔量标记社区标签传播是否发生

            for z in common_neighbors:
                for c in self.g.node[z]['c_coms'].keys():
                    if c in only_v:
                        # v将它与共同邻居z的社区传播给u
                        self.add_to_community(u, c)
                        propagated = True

                    if c in only_u:
                        # u将它与共同邻居z的社区传播给v
                        self.add_to_community(v, c)
                        propagated = True

                for c in shared_coms:
                    # u与v将它们的共同社区传播给共同邻居z
                    if c not in self.g.node[z]['c_coms']:
                        self.add_to_community(z, c)
                        propagated = True

            else:
                if not propagated:
                    # new community
                    # 申请新的社区id
                    actual_cid = self.new_community_id
                    # 该边的两端点加入社区
                    self.add_to_community(u, actual_cid)
                    self.add_to_community(v, actual_cid)

                    # 共同邻居加入社区（组成了三角形）
                    for z in common_neighbors:
                        self.add_to_community(z, actual_cid)

    def print_communities(self):
        """
            Print the actual communities
            打印当前窗口的所有数据（输出到gz压缩文件，生成graph，split，merge和社区划分文件）
            并写入日志
        """
        out_file_coms = gzip.open("%s/%s/strong-communities-%d.gz" % (self.base, self.path, self.actual_slice), "wt", 3)
        com_string = StringIO()

        nodes_to_coms = {}
        merge = {}
        coms_to_remove = []
        drop_c = []

        self.status.write(u"Writing Communities (%s)\n" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()
        for idc, comk in iteritems(self.communities):

            com = comk.keys()

            if self.communities[idc] is not None:
                if len(com) > 2:
                    key = tuple(sorted(com))

                    # Collision check and merge index build (maintaining the lowest id)
                    if key not in nodes_to_coms:
                        nodes_to_coms[key] = idc
                    else:
                        old_id = nodes_to_coms[key]
                        drop = idc
                        if idc < old_id:
                            drop = old_id
                            nodes_to_coms[key] = idc

                        # merged to remove
                        coms_to_remove.append(drop)
                        if not nodes_to_coms[key] in merge:
                            merge[nodes_to_coms[key]] = [idc]
                        else:
                            merge[nodes_to_coms[key]].append(idc)
                else:
                    drop_c.append(idc)
            else:
                drop_c.append(idc)

        write_count = 0

        # 当前时间点节点相似度（距离）可视化
        if self.plot_node_simi:
            self.plot_slice_node_similarity(nodes_to_coms, self.path + f"/nodes_similarity_{self.actual_slice}.jpg")

        coms_cc_lt = []
        for k, idk in iteritems(nodes_to_coms):
            # 计算聚类系数
            nodes_cc_lt = [cc for cc in nx.clustering(self.g, k).values()]
            com_avg_cc = sum(nodes_cc_lt) / len(nodes_cc_lt)
            coms_cc_lt.append(com_avg_cc)

            write_count += 1
            if write_count % 50000 == 0:
                out_file_coms.write(com_string.getvalue())
                out_file_coms.flush()  # 刷新文件缓冲区，将缓冲区的内容提前写入文件
                com_string = StringIO()
                write_count = 0
            com_string.write(u"%d\t%s\n" % (idk, str(list(k))))

        if self.count_avg_cc:
            # 当前时间片整个网络的平均cc
            slice_avg_cc = sum(coms_cc_lt) / len(coms_cc_lt)
            self.slices_cc.append(slice_avg_cc)
            # 将时间片的平均聚类系数写入文件
            with open(self.path + f"/cc-{self.actual_slice}.txt", "w", encoding="utf8") as wf:
                wf.write(str(coms_cc_lt) + "\n")
                wf.write(str(slice_avg_cc) + "\n")

        for dc in drop_c:
            self.destroy_community(dc)

        out_file_coms.write(com_string.getvalue())
        out_file_coms.flush()
        out_file_coms.close()

        # write the graph
        self.status.write(u"Writing actual graph status (%s)\n" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()
        out_file_graph = gzip.open("%s/%s/graph-%d.gz" % (self.base, self.path, self.actual_slice), "wt", 3)
        g_string = StringIO()
        for e in self.g.edges():
            # 输出某个时间点的图结构到文件时，也会带上边的权值
            # g_string.write(u"%d\t%s\t%d\n" % (e[0], e[1], self.g.adj[e[0]][e[1]]['weight']))
            g_string.write("{0}\t{1}\t{2}\n".format(e[0], e[1], self.g.adj[e[0]][e[1]]['weight']))

        out_file_graph.write(g_string.getvalue())
        out_file_graph.flush()
        out_file_graph.close()

        # Write merge status
        self.status.write(u"Writing merging file (%s)\n" % str(time.asctime(time.localtime(time.time()))))
        self.status.flush()
        out_file_merge = gzip.open("%s/%s/merging-%d.gz" % (self.base, self.path, self.actual_slice), "wt", 3)
        m_string = StringIO()
        for comid, c_val in iteritems(merge):
            # maintain minimum community after merge
            c_val.append(comid)
            k = min(c_val)
            c_val.remove(k)
            m_string.write(u"%d\t%s\n" % (k, str(c_val)))
        out_file_merge.write(m_string.getvalue())
        out_file_merge.flush()
        out_file_merge.close()

        # Community Cleaning
        m = 0
        for c in coms_to_remove:
            self.destroy_community(c)
            m += 1

        self.status.write(u"Merged communities: %d (%s)\n" % (m, str(time.asctime(time.localtime(time.time())))))

        self.actual_slice += 1
        self.status.write(u"Total Communities %d (%s)\n" % (len(self.communities.keys()),
                                                            str(time.asctime(time.localtime(time.time())))))
        self.status.flush()

    def destroy_community(self, cid):
        """
        销毁一个社区
        :param cid: 社区id
        :return: None
        """
        # 获得社区中的所有成员点
        nodes = [x for x in self.communities[cid].keys()]
        for n in nodes:
            self.remove_from_community(n, cid)  # 将所有成员节点从该社区中移除
        self.communities.pop(cid, None)  # 从社区列表中移除指定的社区

    def add_to_community(self, node, cid):
        """
        添加一个节点到一个社区
        只处理成员关系，不处理社区更新
        :param node: 节点id
        :param cid: 社区id
        :return: None
        """
        self.g.node[node]['c_coms'][cid] = None
        if cid in self.communities:
            self.communities[cid][node] = None
        else:
            self.communities[cid] = {node: None}

    def remove_from_community(self, node, cid):
        """
        将一个节点从一个社区中移除
        只处理成员关系，不处理社区更新问题
        :param node: 节点id
        :param cid: 社区id
        :return: None
        """
        if cid in self.g.node[node]['c_coms']:
            # 在nx graph对象的节点中存储了某节点所属的社区，需要在这里清除
            self.g.node[node]['c_coms'].pop(cid, None)
            if cid in self.communities and node in self.communities[cid]:
                # 成员对象communities表中也存储了社区的成员节点id，需要清除
                self.communities[cid].pop(node, None)

    def centrality_test(self, subgraph):
        """
        计算一个子图中所有能组成三角形的点（核心点）
        注意：与计算社区不同，即使子图不连通，也符合返回的标准
        源码中使用central表示论文中的核心点（core）
        :param subgraph: 子图
        :return: 节点集合（字典）
        """
        central = {}

        for u in subgraph.nodes():
            if u not in central:
                cflag = False
                neighbors_u = set(self.g.neighbors(u))  # 获取该节点的邻居点
                if len(neighbors_u) > 1:
                    for v in neighbors_u:
                        if u > v:
                            if cflag:
                                break
                            else:
                                neighbors_v = set(self.g.neighbors(v))
                                common_neighbour = neighbors_v & neighbors_v  # 计算共同邻居
                                if len(common_neighbour) > 0:
                                    # 共同邻居>0，则说明这三个点组成了三角形，将他们都加入central
                                    # 在central字典中用None值来标记某个点的归属
                                    central[u] = None
                                    central[v] = None
                                    for n in common_neighbour:
                                        central[n] = None
                                    cflag = True
        return central

    @timer(DEBUG)
    def plot_slice_node_similarity(self, nodes_to_coms, image_dump_path):
        """warning: 在动态节点属性场景中未经测试"""
        heat_map_data = []
        for members, comm_id in tqdm(nodes_to_coms.items()):
            # cur_row = []
            for member in members:
                cur_row = [self.node_attr_handler.get_distance(member, node)
                           for comm_m in nodes_to_coms for node in comm_m]
                # cur_row.append(
                #     [self.node_attr_handler.get_distance(member, node)
                #      for comm_m in nodes_to_coms for node in comm_m])
                heat_map_data.append(cur_row)

        sns.heatmap(heat_map_data)
        plt.xlabel("nodes")
        plt.ylabel("nodes")

        plt.savefig(image_dump_path)
