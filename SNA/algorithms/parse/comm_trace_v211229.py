# -*- coding: utf-8 -*-
# @Time: 2021/3/9 16:40
# @Author: Rollbear
# @Filename: comm_trace_v211229.py

import gzip
import json
import re
import time
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
from SNA.algorithms.parse.comm_select_v211229 import *


class FormatError(Exception):
    pass


class SliceTerminate(Exception):
    pass


class CommTrace:
    def __init__(self, comm_id, tiles_output_dir, screen_name_map_path, parse_relative_sub_comm=False,
                 debug=False, in_sub_comm_trace=False, start_slice_id=0):
        self.comm_id = comm_id
        self.timeline = []
        self.all_timeline = []
        self._data = {}
        self._merge_time = []
        self._split_time = []
        self._parse_relative_sub_comm = parse_relative_sub_comm
        self._tiles_output_dir = tiles_output_dir
        self._debug = debug
        self._in_sub_comm_trace = in_sub_comm_trace
        self._start_slice_id = start_slice_id

        self.network_profile = {}
        self._time_to_break = False

        self._screen_name_map_path = screen_name_map_path
        self._screen_name_map = json.load(open(screen_name_map_path, "r", encoding="utf8"))

        self._real_datetime_map = {}

        self._network_handler = CommSelector(self._tiles_output_dir)

        self._comm_trace()

    def dump_data(self, dump_path):
        json.dump(self._data, open(dump_path, "w", encoding="utf8"), indent=2)

    def dump_viz(self, dump_dir):
        for life_id, time_shot in enumerate(self._data):
            graph = nx.Graph()  # 无向图
            if "edgelist" in self._data[time_shot]:
                for edge in self._data[time_shot]["edgelist"]:
                    graph.add_edge(edge[0], edge[1], weight=edge[2])

            plt.clf()
            pos = nx.spring_layout(graph)
            capacity = nx.get_edge_attributes(graph, "label")

            # 对新增节点和旧节点着色
            pre_slice_nodes = self._data[time_shot - 1]["cur_members"] if life_id != 0 else []
            color_map = ["#a195fb" if node in pre_slice_nodes else "#ff8696" for node in graph.nodes]

            if len(graph.nodes) <= 50:
                nx.draw_networkx_nodes(graph, pos, node_color=color_map)  # 画出点
                nx.draw_networkx_edges(graph, pos)  # 画出边
                nx.draw_networkx_labels(graph, pos)  # 画出点上的label
                nx.draw_networkx_edge_labels(graph, pos, capacity)  # 画出边上的label（例如权）
            else:
                nx.draw_networkx_nodes(graph, pos, node_size=30, node_color=color_map)  # 画出点
                nx.draw_networkx_edges(graph, pos)  # 画出边
                nx.draw_networkx_edge_labels(graph, pos, capacity)  # 画出边上的label（例如权）

            dump_path = dump_dir + self.comm_id + "_" + str(time_shot) + ".jpg"
            plt.savefig(dump_path)

    def dump_community_profile_version2(self, dump_dir: str):
        """2021-04-14: 新的社区元数据格式"""
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        dump_dir = dump_dir if dump_dir.endswith("/") else dump_dir + "/"

        self.dump_events(dump_path=dump_dir + "events.json")

        self.dump_comm_meta_data_version2(out_dir=dump_dir, special_node_mode="merged_comm",
                                          dump_file_name="evo_data.json")
        self.dump_population_evolution_json(out_dir=dump_dir)

    def dump_events(self, dump_path: str):
        """
        0416新数据格式
        [{"datetime":"","event":"","related_comms":"","related_nodes":""},{},...]
        """
        # data = {}
        # for s in self._data:
        #     if "events" in self._data[s]:
        #         data[s] = self._data[s]["events"]

        def get_event_related_nodes(event_type, event_data, s):
            """子函数，处理加载事件相关社区的成员的过程"""
            nodes = set()
            for comm in event_data:
                if event_type == "merging":
                    slices = sorted([s_id for s_id, slice_data in self._network_handler.data[comm].items()
                                     if "cur_members" in slice_data and len(slice_data["cur_members"]) != 0])
                    if len(slices) == 0:
                        continue
                    last_s = slices[-1]
                    for node in self._network_handler.data[comm][last_s]["cur_members"]:
                        nodes.add(node)
                else:
                    if s in self._network_handler.data[comm] \
                            and "cur_members" in self._network_handler.data[comm][s]:
                        for node in self._network_handler.data[comm][s]["cur_members"]:
                            nodes.add(node)
            return list(nodes)

        data = [
            {
                "datetime": self._real_datetime_map[s].split(" ")[4]
                if s in self._real_datetime_map else "",
                "events": [
                    {
                        "type": event_type,
                        "related_comms": event_data,  # 相关的社区id的列表
                        # 相关社区的成员，去重
                        "related_nodes": get_event_related_nodes(event_type, event_data, s)
                    } for event_type, event_data in self._data[s]["events"].items()
                ]
            }
            for s in self._data if "events" in self._data[s]
        ]

        with open(dump_path, "w", encoding="utf8") as wf:
            json.dump(data, wf, indent=2)

    def _comm_trace(self):
        cur_slice_id = self._start_slice_id

        while True:
            try:
                if self._debug and not self._in_sub_comm_trace:
                    print(f"\r[INFO] in CommTracer, current slice: {cur_slice_id}")
                self._trace_strong_comm(cur_slice_id)
                self._trace_merging(cur_slice_id)
                self._trace_splitting(cur_slice_id)
                self._trace_graph(cur_slice_id)

                cur_slice_id += 1

            except FileNotFoundError:
                break
            except SliceTerminate:  # early terminate机制
                break

        self._parse_extract_log(self._tiles_output_dir)

    def _trace_strong_comm(self, cur_slice_id):
        strong_comm_path = self._tiles_output_dir + f"strong-communities-{cur_slice_id}.gz"
        cur_slice_content_spec_comm = False

        with gzip.open(strong_comm_path, 'rb') as f:
            content = f.read().decode('utf-8')
            for line in content.split("\n"):
                if line != "":
                    arr = line.split("\t")

                    # 追踪所有社区的生命线
                    if arr[0] in self.network_profile:
                        self.network_profile[arr[0]].append(cur_slice_id)
                    else:
                        self.network_profile[arr[0]] = [cur_slice_id]

                    # 追踪特定id的社区
                    if self.comm_id == arr[0]:
                        cur_slice_content_spec_comm = True
                        self.timeline.append(cur_slice_id)
                        cur_members = list(arr[1][1:-1].split(", "))
                        if cur_slice_id in self._data:
                            self._data[cur_slice_id]["cur_members"] = cur_members
                        else:
                            self._data[cur_slice_id] = {"cur_members": cur_members}

                        # 加上账号昵称的字段，方便查看
                        self._data[cur_slice_id]["member_names"] = \
                            [self._screen_name_map[m] if m in self._screen_name_map else "unknown" for m in cur_members]

            # 提前终止机制，提高解析效率
            if not cur_slice_content_spec_comm:
                if len(self._data) > 0:  # 若len = 0，说明还没到该社区出现的时间点
                    self._time_to_break = True  # 在本时间片解析完毕后提前退出
                    print(f"[INFO] comm {self.comm_id}, not appear in a slice, early terminated")

        self.all_timeline.append(cur_slice_id)

    def _trace_merging(self, cur_slice_id):
        merging_path = self._tiles_output_dir + f"merging-{cur_slice_id}.gz"
        with gzip.open(merging_path, "rb") as f:
            content = f.read().decode("utf8")
            for line in content.split("\n"):
                if line != "":
                    arr = line.split("\t")
                    master = arr[0]
                    merged_lt = list(arr[1][1:-1].split(", "))

                    # 当前社区是合并者
                    if self.comm_id == master:
                        self._merge_time.append(cur_slice_id)

                        # 如果设置了_parse_relative_sub_comm，则解析具有一阶关联的其他社区
                        if self._parse_relative_sub_comm:
                            tmp = []
                            for comm in merged_lt:
                                start = time.clock()  # timer
                                print(
                                    f"[TIME] parent comm in slice {cur_slice_id}, relative comm {comm}, tracing start.")  # log

                                tracer = CommTrace(comm_id=comm, parse_relative_sub_comm=False,
                                                   tiles_output_dir=self._tiles_output_dir, debug=self._debug,
                                                   in_sub_comm_trace=True,
                                                   screen_name_map_path=self._screen_name_map_path)  # 只解析一阶关联社区，慎防无限递归
                                comm_data = tracer._data
                                tmp.append({"comm_id": comm, "data": comm_data})

                                end = time.clock()  # timer
                                print(
                                    f"[TIME] parent comm in slice {cur_slice_id}, relative comm: {comm}, trace in {end - start}s.")  # log
                            merged_lt = tmp

                        if cur_slice_id in self._data:
                            if "events" in self._data[cur_slice_id]:
                                if "merging" in self._data[cur_slice_id]["events"]:
                                    self._data[cur_slice_id]["events"]["merging"] += merged_lt
                                else:
                                    self._data[cur_slice_id]["events"]["merging"] = merged_lt
                            else:
                                self._data[cur_slice_id]["events"] = {"merging": merged_lt}
                        else:
                            self._data[cur_slice_id] = {"events": {"merging": merged_lt}}

                    # 当前社区是被合并者
                    elif self.comm_id in merged_lt:
                        self._merge_time.append(cur_slice_id)
                        if cur_slice_id in self._data:
                            if "events" in self._data[cur_slice_id]:
                                if "merged/died" in self._data[cur_slice_id]["events"]:
                                    self._data[cur_slice_id]["events"]["merged/died"].append(master)
                                else:
                                    self._data[cur_slice_id]["events"]["merged/died"] = [master]
                            else:
                                self._data[cur_slice_id]["events"] = {"merged/died": [master]}
                        else:
                            self._data[cur_slice_id] = {"events": {"merged/died": [master]}}
                        self._time_to_break = True  # early terminate机制，如果该社区已经分裂，说明后续的时间点不会再有这个社区id了
                        print(f"[INFO] comm {self.comm_id} merged, early terminated.")  # log

    def _trace_graph(self, cur_slice_id):
        graph_path = self._tiles_output_dir + f"graph-{cur_slice_id}.gz"
        with gzip.open(graph_path, "rb") as f:
            content = f.read().decode("utf8")
            for line in content.split("\n"):
                if line != "":
                    arr = line.split("\t")
                    source = arr[0]
                    dest = arr[1]
                    weight = arr[2]

                    if cur_slice_id in self._data:
                        if "cur_members" in self._data[cur_slice_id]:
                            if source in self._data[cur_slice_id]["cur_members"] \
                                    and dest in self._data[cur_slice_id]["cur_members"]:
                                if "edgelist" in self._data[cur_slice_id]:
                                    self._data[cur_slice_id]["edgelist"].append((source, dest, weight))
                                else:
                                    self._data[cur_slice_id]["edgelist"] = [(source, dest, weight)]
        if self._time_to_break:
            # 把社区消亡也计入events
            self._data[cur_slice_id] = {"events": {"died": []}}
            raise SliceTerminate

    def _trace_splitting(self, cur_slice_id):
        splitting_path = self._tiles_output_dir + f"splitting-{cur_slice_id}.gz"
        with gzip.open(splitting_path, "rb") as f:
            content = f.read().decode("utf8")
            for line in content.split("\n"):
                if line != "":
                    arr = line.split("\t")
                    splitting_comm = arr[0]
                    new_comm_lt = list(arr[1][1:-1].split(", "))

                    if self.comm_id == splitting_comm:
                        self._split_time.append(cur_slice_id)

                        # 如果设置了_parse_relative_sub_comm，则解析具有一阶关联的其他社区
                        if self._parse_relative_sub_comm:
                            tmp = []
                            for comm in new_comm_lt:
                                start = time.clock()  # timer
                                print(
                                    f"[TIME] parent comm in slice {cur_slice_id}, relative comm {comm}, tracing start.")  # log

                                tracer = CommTrace(comm_id=comm, parse_relative_sub_comm=False,
                                                   tiles_output_dir=self._tiles_output_dir, debug=self._debug,
                                                   in_sub_comm_trace=True, start_slice_id=cur_slice_id - 1,
                                                   screen_name_map_path=self._screen_name_map_path)  # 只解析一阶关联社区，慎防无限递归
                                comm_data = tracer._data
                                tmp.append({"comm_id": comm, "data": comm_data})

                                end = time.clock()  # timer
                                print(
                                    f"[TIME] parent comm in slice {cur_slice_id}, relative comm: {comm}, trace in {end - start}s.")  # log

                            new_comm_lt = tmp

                        if cur_slice_id in self._data:
                            if "events" in self._data[cur_slice_id]:
                                if "splitting/died" in self._data[cur_slice_id]["events"]:
                                    self._data[cur_slice_id]["events"]["splitting/died"] += new_comm_lt
                                else:
                                    self._data[cur_slice_id]["events"]["splitting/died"] = new_comm_lt
                            else:
                                self._data[cur_slice_id]["events"] = {"splitting/died": new_comm_lt}
                        else:
                            self._data[cur_slice_id] = {"events": {"splitting/died": new_comm_lt}}
                        self._time_to_break = True  # early terminate机制，如果该社区已经分裂，说明后续的时间点不会再有这个社区id了
                        print(f"[INFO] comm {self.comm_id} splitting, early terminated.")  # log

                    if self.comm_id in new_comm_lt:
                        print(f"[ERROR] comm {self.comm_id}, appear in both splitting comm and new comm.")
                        # raise FormatError

    def _parse_extract_log(self, tiles_output_dir):
        pattern = re.compile(r'Saving Slice (\d+): (Starting \d+-\d+-\d+ \d+:\d+:\d+ ending \d+-\d+-\d+ \d+:\d+:\d+) -')
        res = pattern.findall(open(tiles_output_dir + "extraction_status.txt", "r").read())
        for arr in res:
            if int(arr[0]) in self._data:
                self._data[int(arr[0])]["datetime"] = arr[1]

            self._real_datetime_map[int(arr[0])] = arr[1]

    def long_live_comms(self):
        comms_lt = [(comm_id, live_line) for comm_id, live_line in self.network_profile.items()]
        comms_lt.sort(reverse=True, key=lambda elem: len(elem[1]))
        return comms_lt

    def plot_population_evolution(self, dump_path=None):
        plt.clf()  # 清空缓冲
        plt.title('community population', fontsize=20)  # 标题
        plt.xlabel(u'slice id', fontsize=14)  # 设置x轴
        plt.ylabel(u'population (num of nodes)', fontsize=14)  # 设置y轴

        plt.plot([s_id for s_id in self._data.keys() if "cur_members" in self._data[s_id]],
                 [len(s["cur_members"]) for s in self._data.values() if "cur_members" in s])
        if dump_path:
            plt.savefig(dump_path)
        else:
            plt.show()

    def plot_early_slice(self, random_neighbors=None, show_node_label=True,
                         dump_to_gexf=None, mode="neighbors", dump_to_json=None, return_json_data=False):
        plt.clf()
        slice_id = list(self._data.keys())[0] - 1  # 社区诞生的前一个观测点
        early_members = self._data[slice_id + 1]["cur_members"]

        sub_graph_nodes = [] + early_members
        graph_path = self._tiles_output_dir + f"graph-{slice_id}.gz"
        graph = nx.Graph()

        try:
            with gzip.open(graph_path, "rb") as f:
                content = f.read().decode("utf8")
                for line in tqdm(content.split("\n"), desc="loading graph: "):
                    if line != "":
                        arr = line.split("\t")
                        source = arr[0]
                        dest = arr[1]
                        weight = arr[2]
                        graph.add_edge(source, dest, weight=weight)

            # 加载的节点为一阶邻居
            if mode == "neighbors":
                # 一阶邻居在子图中显示
                for node in sub_graph_nodes:
                    if node in graph.nodes:
                        if random_neighbors is None:
                            neighbors = list(graph.neighbors(node))
                        else:
                            neighbors = random.sample(list(graph.neighbors(node)), random_neighbors)
                        sub_graph_nodes = list(set(sub_graph_nodes + neighbors))

            # 加载的节点为涉及的社区的成员点
            elif mode == "community":
                # 在社区诞生的前一个时间点中遍历社区，找出那些包含了early members的社区
                strong_comm_path = self._tiles_output_dir + f"strong-communities-{slice_id}.gz"

                with gzip.open(strong_comm_path, 'rb') as f:
                    content = f.read().decode('utf-8')
                    for line in content.split("\n"):
                        if line != "":
                            arr = line.split("\t")
                            # comm_id = arr[0]
                            cur_members = list(arr[1][1:-1].split(", "))

                            # 若early members是某个社区的成员，将这个社区的成员加入
                            for early_member in early_members:
                                if early_member in cur_members:
                                    sub_graph_nodes += cur_members
                sub_graph_nodes = list(set(sub_graph_nodes))  # 去重

            # 标签那些成员点
            for node_id in tqdm(list(graph.nodes), desc="labeling nodes: "):
                if node_id in early_members:
                    graph.add_node(node_id, is_member=1)
                else:
                    graph.add_node(node_id, is_member=0)

            sub_graph = graph.subgraph(sub_graph_nodes)

            if not dump_to_gexf and not dump_to_json and not return_json_data:
                plt.clf()
                pos = nx.spring_layout(sub_graph)
                capacity = nx.get_edge_attributes(sub_graph, "label")

                color_map = ["#a195fb" if node in early_members else "#ff8696" for node in sub_graph_nodes]
                nx.draw_networkx_nodes(sub_graph, pos, node_color=color_map)  # 画出点
                nx.draw_networkx_edges(sub_graph, pos)  # 画出边
                if show_node_label:
                    nx.draw_networkx_labels(sub_graph, pos)  # 画出点上的label
                nx.draw_networkx_edge_labels(sub_graph, pos, capacity)  # 画出边上的label（例如权）

                plt.show()

            if dump_to_gexf:
                nx.write_gexf(sub_graph, dump_to_gexf)

            special_nodes = [node for node in sub_graph_nodes if node not in early_members]

            out_data = {
                "slice_id": slice_id,
                "datetime_interval": self._real_datetime_map[slice_id].split(" ")[4]
                if slice_id in self._real_datetime_map else "",
                "screen_name_map": [{"id": n, "screen_name": self._screen_name_map.get(n, "unknown")}
                                    for n in sub_graph_nodes],
                "edge_list": [{"source": edge[0], "target": edge[1]} for edge in sub_graph.edges],
                "special_nodes": special_nodes
            }

            if dump_to_json:
                with open(dump_to_json, "w", encoding="utf8") as wf:
                    json.dump(out_data, wf, indent=2)
            if return_json_data:
                return out_data

        except FileNotFoundError:
            pass

    def plot_later_slice(self, random_neighbors=None, show_node_label=True,
                         dump_to_gexf=None, mode="neighbors", dump_to_json=None, return_json_data=False):
        plt.clf()
        final_slice = list(self._data.keys())[-1]  # 消失后的第一个观测点
        if "cur_members" in self._data[final_slice]:
            slice_id = final_slice + 1
            final_members = self._data[final_slice]["cur_members"]
        else:
            slice_id = final_slice
            final_members = self._data[final_slice - 1]["cur_members"]

        sub_graph_nodes = [] + final_members
        graph_path = self._tiles_output_dir + f"graph-{slice_id}.gz"
        graph = nx.Graph()

        try:
            with gzip.open(graph_path, "rb") as f:
                content = f.read().decode("utf8")
                for line in tqdm(content.split("\n"), desc="loading graph: "):
                    if line != "":
                        arr = line.split("\t")
                        source = arr[0]
                        dest = arr[1]
                        weight = arr[2]
                        graph.add_edge(source, dest, weight=weight)

            if mode == "neighbors":
                # 一阶邻居在子图中显示
                for node in sub_graph_nodes:
                    if node in graph.nodes:
                        if random_neighbors is None:
                            neighbors = list(graph.neighbors(node))
                        else:
                            neighbors = random.sample(list(graph.neighbors(node)), random_neighbors)
                        sub_graph_nodes = list(set(sub_graph_nodes + neighbors))
            elif mode == "community":
                # 显示社区消失后原来的成员点属于的社区
                strong_comm_path = self._tiles_output_dir + f"strong-communities-{slice_id}.gz"

                with gzip.open(strong_comm_path, 'rb') as f:
                    content = f.read().decode('utf-8')
                    for line in content.split("\n"):
                        if line != "":
                            arr = line.split("\t")
                            # comm_id = arr[0]
                            cur_members = list(arr[1][1:-1].split(", "))

                            # 若final members是某个社区的成员，将这个社区的成员加入
                            for early_member in final_members:
                                if early_member in cur_members:
                                    sub_graph_nodes += cur_members
                sub_graph_nodes = list(set(sub_graph_nodes))  # 去重

            for node_id in tqdm(list(graph.nodes), desc="labeling nodes: "):
                if node_id in final_members:
                    graph.add_node(node_id, is_member=1)
                else:
                    graph.add_node(node_id, is_member=0)

            sub_graph = graph.subgraph(sub_graph_nodes)

            if not dump_to_gexf and not dump_to_json and not return_json_data:
                plt.clf()
                pos = nx.spring_layout(sub_graph)
                capacity = nx.get_edge_attributes(sub_graph, "label")

                color_map = ["#a195fb" if node in final_members else "#ff8696" for node in sub_graph_nodes]
                nx.draw_networkx_nodes(sub_graph, pos, node_color=color_map)  # 画出点
                nx.draw_networkx_edges(sub_graph, pos)  # 画出边
                if show_node_label:
                    nx.draw_networkx_labels(sub_graph, pos)  # 画出点上的label
                nx.draw_networkx_edge_labels(sub_graph, pos, capacity)  # 画出边上的label（例如权）

                plt.show()

            if dump_to_gexf:
                nx.write_gexf(sub_graph, dump_to_gexf)

            special_nodes = [node for node in sub_graph_nodes if node not in final_members]

            out_data = {
                "slice_id": slice_id,
                "datetime_interval": self._real_datetime_map[slice_id].split(" ")[4]
                if slice_id in self._real_datetime_map else "",
                "screen_name_map": [{"id": n, "screen_name": self._screen_name_map.get(n, "unknown")}
                                    for n in sub_graph_nodes],
                "edge_list": [{"source": edge[0], "target": edge[1]} for edge in sub_graph.edges],
                "special_nodes": special_nodes
            }

            if dump_to_json:
                with open(dump_to_json, "w", encoding="utf8") as wf:
                    json.dump(out_data, wf, indent=2)
            if return_json_data:
                return out_data

        except FileNotFoundError:
            pass

    def dump_front_end_json(self, out_dir, special_node_mode="new_comer"):
        """将一个社区的数据存储为前端的json交换格式"""
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for s_id, s_data in self._data.items():
            if "cur_members" in s_data:  # 不存储最后一个时间片
                out_path = out_dir + f"{self.comm_id}_{s_id}.json"
                out_data = self._get_timeline_data_single_slice(s_id, special_node_mode)
                json.dump(out_data, open(out_path, "w", encoding="utf8"), indent=2)

    def _get_timeline_data_single_slice(self, s_id: int, special_node_mode) -> dict:
        special_nodes = []
        s_data = self._data[s_id]
        if special_node_mode == "new_comer":
            # special nodes这里指该时间片新加入的节点
            pre_slice_nodes = self._data[s_id - 1]["cur_members"] if s_id != list(self._data.keys())[0] else []
            special_nodes = [node for node in s_data["cur_members"] if node not in pre_slice_nodes]
        elif special_node_mode == "merged_comm":
            # 这里指发生了合并事件的点
            if "events" in s_data:
                for event, involves in s_data["events"].items():
                    # special_nodes += involves  # todo::involves应当是节点而不是社区
                    for comm in involves:
                        slices = sorted([s for s, slice_data in self._network_handler.data[comm].items()
                                         if "cur_members" in slice_data and len(slice_data["cur_members"]) != 0])
                        if len(slices) == 0:
                            continue
                        last_slice = slices[-1]
                        merging_members = self._network_handler.data[comm][last_slice]["cur_members"]
                        special_nodes += [member for member in merging_members if member in s_data["cur_members"]]
        else:
            raise Exception

        # 去重
        special_nodes = list(set(special_nodes))

        out_data = {
            "slice_id": s_id,
            # 0407新增：生成json数据时截断时间区间字符串，仅保留时间片结束日期（精确到日）
            "datetime_interval": s_data["datetime"].split(" ")[4] if "datetime" in s_data else "",
            "screen_name_map": [{"id": n, "screen_name": self._screen_name_map.get(n, "unknown")}
                                for n in s_data["cur_members"]],
            "edge_list": [{"source": edge[0], "target": edge[1]}
                          for edge in s_data["edgelist"]] if "edgelist" in s_data else [],
            "special_nodes": special_nodes
        }
        return out_data

    # def dump_comm_meta_data(self, out_dir):
    #     """存储社区的元数据，元数据样例如下
    #     {
    #         comm_id: "123",
    #         timeline: [10, 11, 12, 13, ...],
    #         timeline_count: 10,
    #         real_datetime_map: [
    #             {slice_id: 10, datetime: "2012-03-18"},
    #             {slice_id: 11, datetime: "2012-04-18"},
    #             ...
    #         ]
    #     }"""
    #     out_dir = out_dir if out_dir.endswith("/") else out_dir + "/"
    #     avail_slices = [k for k in self._data.keys() if "cur_members" in self._data[k]]
    #
    #     # 构造键值对数据
    #     data = {
    #         "comm_id": self.comm_id,
    #         "timeline": avail_slices,
    #         "timeline_count": len(avail_slices),
    #         "real_datetime_map": [{"slice_id": s, "datetime": self._real_datetime_map[s].split(" ")[4] if s in self._real_datetime_map else "unknown"}
    #                               for s in avail_slices]
    #     }
    #
    #     # 输出到json文件
    #     out_path = out_dir + "comm_meta_data.json"
    #     with open(out_path, "w", encoding="utf8") as wf:
    #         json.dump(data, wf, indent=2)

    def dump_comm_meta_data_version2(self, out_dir, special_node_mode, dump_file_name):
        """0414新数据格式"""
        out_dir = out_dir if out_dir.endswith("/") else out_dir + "/"
        avail_slices = [k for k in self._data.keys() if "cur_members" in self._data[k]]

        # 构造键值对数据
        data = {
            "comm_id": self.comm_id,
            "timeline_datetime": [self._real_datetime_map[s].split(" ")[4]
                                  if s in self._real_datetime_map else "unknown"
                                  for s in avail_slices],
            "timeline_count": len(avail_slices),
            # timeline_data里面每个元素是原timeline的一个json文件的内容（单个时间片的数据）
            "timeline_data": [self._get_timeline_data_single_slice(s_id=s, special_node_mode=special_node_mode)
                              for s in avail_slices]
        }
        # timeline data中加入新生前一个时间片和消亡后一个时间片
        data["timeline_data"].insert(0, self.plot_early_slice(mode="community", return_json_data=True))
        data["timeline_data"].append(self.plot_later_slice(mode="community", return_json_data=True))

        # 输出到json文件
        out_path = out_dir + dump_file_name
        with open(out_path, "w", encoding="utf8") as wf:
            json.dump(data, wf, indent=2)

    def dump_population_evolution_json(self, out_dir):
        """存储社区规模变化数据，json格式，样例如下
        [
            {
                slice_id: 8,
                datetime_interval: "Starting 2012-03-09 08:05:12 ending 2012-03-18 04:30:58",
                value: 500
            },
            {
                slice_id: 9,
                datetime_interval: "Starting 2012-03-09 08:05:12 ending 2012-03-18 04:30:58",
                value: 600
            },
            ......
        ]
        """
        out_dir = out_dir if out_dir.endswith("/") else out_dir + "/"

        out_data = [
            {
                "slice_id": s,
                "datetime_interval": self._real_datetime_map[s].split(" ")[4]
                if s in self._real_datetime_map else "unknown",
                "value": len(self._data[s]["cur_members"])
            }
            for s in self._data if "cur_members" in self._data[s]
        ]
        out_path = out_dir + "population_evo.json"
        with open(out_path, "w", encoding="utf8") as wf:
            json.dump(out_data, wf, indent=2)


def main():
    pass


if __name__ == '__main__':
    main()
