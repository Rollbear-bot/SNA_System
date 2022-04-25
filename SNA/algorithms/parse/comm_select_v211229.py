# -*- coding: utf-8 -*-
# @Time: 2021/3/11 8:28
# @Author: Rollbear
# @Filename: comm_select_v211229.py
# 从tiles算法的输出中挑选社区

import gzip
import json
import re
# from multiprocessing import Process
from tqdm import tqdm


class CommSelector:
    def __init__(self, tiles_output_dir, load_elt=False):
        self._data = {}
        self.tiles_output_dir = tiles_output_dir
        self._load_elt = load_elt
        self._comm_count = {}  # 社区数量的统计

        # self._events_count = {}  # 事件数量的统计，仅统计分裂和合并事件
        self.merge_count = {}
        self.split_count = {}
        self.birth_count = {}
        self.death_count = {}

        self.birth_records = {}
        self.death_records = {}
        self.merge_records = {}
        self.split_records = {}

        # 节点和边的数量增长统计（在读取graph文件时得到）
        self.node_count = {}
        self.edge_count = {}

        self._real_datetime_map = {}  # 时间片到具体时间的映射

        self._init_data()

    @property
    def real_datetime_map(self):
        return self._real_datetime_map

    @property
    def comm_count(self):
        return self._comm_count

    @property
    def data(self):
        return self._data

    def _init_data(self):
        # 先加载社区，才能加载其他
        self._init_strong_comm()
        print("[INFO] load strong comm done.")
        self._init_other()
        print("[INFO] load other done.")

    def _init_strong_comm(self):
        cur_slice_id = 0  # tiles输出的时间片id从0开始
        while True:
            try:
                self._load_strong_comm(cur_slice_id)
                cur_slice_id += 1
            except FileNotFoundError:
                break

    def _init_other(self):
        cur_slice_id = 0
        while True:
            try:
                self._load_merge(cur_slice_id)
                self._load_split(cur_slice_id)
                if self._load_elt:
                    self._load_graph(cur_slice_id)
                cur_slice_id += 1

            except FileNotFoundError:
                break
        self._load_extract_log()

    # ==========================================================================
    # 解析函数
    # ==========================================================================

    def _load_strong_comm(self, cur_slice_id):
        strong_comm_path = self.tiles_output_dir + f"strong-communities-{cur_slice_id}.gz"
        comm_count = 0  # 该时间片的社区计数
        cur_slice_comms = set()

        with gzip.open(strong_comm_path, 'rb') as f:
            content = f.read().decode('utf-8')
            for line in content.split("\n"):
                if line != "":
                    comm_count += 1
                    arr = line.split("\t")
                    comm_id = arr[0]
                    cur_slice_comms.add(comm_id)
                    cur_members = list(arr[1][1:-1].split(", "))
                    if comm_id in self._data:
                        if cur_slice_id in self._data[comm_id]:
                            if "cur_members" in self._data[comm_id][cur_slice_id]:
                                self._data[comm_id][cur_slice_id]["cur_members"] += cur_members
                            else:
                                self._data[comm_id][cur_slice_id]["cur_members"] = cur_members
                        else:
                            self._data[comm_id][cur_slice_id] = {"cur_members": cur_members}
                    else:
                        self._data[comm_id] = {cur_slice_id: {"cur_members": cur_members}}

            # 加载当前时间片的社区完毕后，统计新生的（当前含而上一时间不含的）和消亡（上一时间含而当前时间不含的）的社区数量
            if cur_slice_id == 0:
                self.birth_count[cur_slice_id] = 0
                self.death_count[cur_slice_id] = 0
            else:
                pre_slice_comms = {comm for comm in self._data if cur_slice_id - 1 in self._data[comm]}
                birth_comms = {comm for comm in cur_slice_comms if comm not in pre_slice_comms}
                death_comms = {comm for comm in pre_slice_comms if comm not in cur_slice_comms}
                self.birth_count[cur_slice_id] = len(birth_comms)
                self.birth_records[cur_slice_id] = birth_comms.copy()
                self.death_count[cur_slice_id] = len(death_comms)
                self.death_records[cur_slice_id] = death_comms.copy()

        self._comm_count[cur_slice_id] = comm_count

    def _load_merge(self, cur_slice_id):
        merging_path = self.tiles_output_dir + f"merging-{cur_slice_id}.gz"
        merge_count = 0  # 该时间片merge事件的计数

        with gzip.open(merging_path, "rb") as f:
            content = f.read().decode("utf8")
            for line in content.split("\n"):
                if line != "":
                    merge_count += 1
                    arr = line.split("\t")
                    master = arr[0]
                    merged_lt = list(arr[1][1:-1].split(", "))

                    # 保存合并者
                    if master in self._data:
                        if cur_slice_id in self._data[master]:
                            if "events" in self._data[master][cur_slice_id]:
                                if "merging" in self._data[master][cur_slice_id]["events"]:
                                    self._data[master][cur_slice_id]["events"]["merging"] += merged_lt
                                else:
                                    self._data[master][cur_slice_id]["events"]["merging"] = merged_lt
                            else:
                                self._data[master][cur_slice_id]["events"] = {"merging": merged_lt}
                        else:
                            self._data[master][cur_slice_id] = {"events": {"merging": merged_lt}}
                    else:
                        self._data[master] = {cur_slice_id: {"events": {"merging": merged_lt}}}

                    # 保存被合并者
                    for comm in merged_lt:
                        if comm in self._data:
                            if cur_slice_id in self._data[comm]:
                                if "events" in self._data[comm][cur_slice_id]:
                                    if "merged/died" in self._data[comm][cur_slice_id]["events"]:
                                        self._data[comm][cur_slice_id]["events"]["merged/died"].append(master)
                                    else:
                                        self._data[comm][cur_slice_id]["events"]["merged/died"] = [master]
                                else:
                                    self._data[comm][cur_slice_id]["events"] = {"merged/died": [master]}
                            else:
                                self._data[comm][cur_slice_id] = {"events": {"merged/died": [master]}}
                        else:
                            self._data[comm] = {cur_slice_id: {"events": {"merged/died": [master]}}}

                    # 保存合并转捩点
                    if cur_slice_id not in self.merge_records:
                        self.merge_records[cur_slice_id] = [(master, merged_lt)]
                    else:
                        self.merge_records[cur_slice_id].append((master, merged_lt))

        if cur_slice_id in self.merge_count:
            self.merge_count[cur_slice_id] += merge_count
        else:
            self.merge_count[cur_slice_id] = merge_count

    def _load_split(self, cur_slice_id):
        splitting_path = self.tiles_output_dir + f"splitting-{cur_slice_id}.gz"
        split_count = 0  # 分裂事件计数

        with gzip.open(splitting_path, "rb") as f:
            content = f.read().decode("utf8")
            for line in content.split("\n"):
                if line != "":
                    split_count += 1
                    arr = line.split("\t")
                    splitting_comm = arr[0]
                    new_comm_lt = list(arr[1][1:-1].split(", "))

                    # 保存分裂的社区
                    if splitting_comm in self._data:
                        if cur_slice_id in self._data[splitting_comm]:
                            if "events" in self._data[splitting_comm][cur_slice_id]:
                                if "splitting" in self._data[splitting_comm][cur_slice_id]["events"]:
                                    self._data[splitting_comm][cur_slice_id]["events"]["splitting"] += new_comm_lt
                                else:
                                    self._data[splitting_comm][cur_slice_id]["events"]["splitting"] = new_comm_lt
                            else:
                                self._data[splitting_comm][cur_slice_id]["events"] = {"splitting": new_comm_lt}
                        else:
                            self._data[splitting_comm][cur_slice_id] = {"events": {"splitting": new_comm_lt}}
                    else:
                        self._data[splitting_comm] = {cur_slice_id: {"events": {"splitting": new_comm_lt}}}

                    # 保存分裂产生的新社区
                    for comm in new_comm_lt:
                        if comm in self._data:
                            if cur_slice_id in self._data[comm]:
                                if "events" in self._data[comm][cur_slice_id]:
                                    if "generated-by-split" in self._data[comm][cur_slice_id]["events"]:
                                        self._data[comm][cur_slice_id]["events"]["generated-by-split"].append(
                                            splitting_comm)
                                    else:
                                        self._data[comm][cur_slice_id]["events"]["generated-by-split"] = [
                                            splitting_comm]
                                else:
                                    self._data[comm][cur_slice_id]["events"] = {
                                        "generated-by-split": [splitting_comm]}
                            else:
                                self._data[comm][cur_slice_id] = {
                                    "events": {"generated-by-split": [splitting_comm]}}
                        else:
                            self._data[comm] = {cur_slice_id: {"events": {"generated-by-split": [splitting_comm]}}}

                    # 保存分裂转捩点
                    if cur_slice_id not in self.split_records:
                        self.split_records[cur_slice_id] = [(splitting_comm, new_comm_lt)]
                    else:
                        self.split_records[cur_slice_id].append((splitting_comm, new_comm_lt))

        if cur_slice_id in self.split_count:
            self.split_count[cur_slice_id] += split_count
        else:
            self.split_count[cur_slice_id] = split_count

    def _load_graph(self, cur_slice_id):
        graph_path = self.tiles_output_dir + f"graph-{cur_slice_id}.gz"
        node_set = set()
        edge_set = set()

        with gzip.open(graph_path, "rb") as f:
            content = f.read().decode("utf8")

            for line in tqdm(content.split("\n"), desc=f"[TQDM] slice {cur_slice_id} loading graph: "):
                if line != "":
                    arr = line.split("\t")
                    source = arr[0]
                    dest = arr[1]
                    weight = arr[2]

                    # 统计节点数和边数
                    node_set.add(source)
                    node_set.add(dest)
                    edge_set.add((source, dest))

                    for comm in self._data:
                        if cur_slice_id in self._data[comm] and "cur_members" in self._data[comm][cur_slice_id]:
                            if source in self._data[comm][cur_slice_id]["cur_members"] \
                                    or dest in self._data[comm][cur_slice_id]["cur_members"]:
                                if "edgelist" in self._data[comm][cur_slice_id]:
                                    self._data[comm][cur_slice_id]["edgelist"].append(
                                        {"source": source, "dest": dest, "weight": weight})
                                else:
                                    self._data[comm][cur_slice_id]["edgelist"] = \
                                        [{"source": source, "dest": dest, "weight": weight}]
        # 存储该时间片的节点统计和边统计
        self.node_count[cur_slice_id] = len(node_set)
        self.edge_count[cur_slice_id] = len(edge_set)

    def _load_extract_log(self):
        pattern = re.compile(r'Saving Slice (\d+): (Starting \d+-\d+-\d+ \d+:\d+:\d+ ending \d+-\d+-\d+ \d+:\d+:\d+) -')
        res = pattern.findall(open(self.tiles_output_dir + "extraction_status.txt", "r").read())
        for arr in res:
            self._real_datetime_map[int(arr[0])] = arr[1]

    # ==========================================================================
    # 筛选函数
    # ==========================================================================

    def most_long_live_comms(self, top=None, source_keys=None, return_keys=False):
        # 缺省参数
        source_keys = self._data.keys() if source_keys is None else source_keys
        top = len(source_keys) if top is None else top

        data = {key: self._data[key] for key in source_keys}
        keys = sorted(data, key=lambda elem: len(data[elem]), reverse=True)[:top]
        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def most_merging(self, top=None, source_keys=None, return_keys=False):
        # 缺省参数
        source_keys = self._data.keys() if source_keys is None else source_keys
        top = len(source_keys) if top is None else top

        data = {key: self._data[key] for key in source_keys}
        keys = sorted(
            data,
            key=lambda elem: len([data[elem][s]["events"]["merging"] for s in data[elem]
                                  if "events" in data[elem][s] and "merging" in data[elem][s]["events"]]),
            reverse=True
        )[:top]
        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def most_merge_involve_members(self, top=None, source_keys=None, return_keys=False):
        # 缺省参数
        source_keys = self._data.keys() if source_keys is None else source_keys
        top = len(source_keys) if top is None else top
        data = {key: self._data[key] for key in source_keys}

        # 用一个字段记录某个社区是否在生命线内是否发生了分裂事件
        merge_tag = {}
        for key in data:
            for s in data[key]:
                if "events" in data[key][s] and "merging" in data[key][s]["events"]:
                    merge_tag[key] = True
                    break
                merge_tag[key] = False

        # 按照合并后引入的新人数从大到小排序
        keys = sorted(
            data,
            key=lambda elem: max([
                len(data[elem][s]["cur_members"]) - len(data[elem][s - 1]["cur_members"])
                if s-1 in data[elem] else 0
                for s in data[elem]
                if "events" in data[elem][s] and "merging" in data[elem][s]["events"]
            ]) if merge_tag[elem] else 0,
            reverse=True
        )[:top]

        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def most_splitting(self, top=None, source_keys=None, return_keys=False):
        # 缺省参数
        source_keys = self._data.keys() if source_keys is None else source_keys
        top = len(source_keys) if top is None else top

        data = {key: self._data[key] for key in source_keys}
        keys = sorted(
            data,     # 20210821改动：选择分裂出社区最多的，而不是分裂次数
            key=lambda elem: sum([len(data[elem][s]["events"]["splitting"]) for s in data[elem]
                                  if "events" in data[elem][s] and "splitting" in data[elem][s]["events"]]),
            reverse=True
        )[:top]
        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def born_from_split(self, source_keys=None, return_keys=False):
        source_keys = self._data.keys() if source_keys is None else source_keys
        data = {key: self._data[key] for key in source_keys}
        keys = [key for key in data
                if len([s for s in data[key] if
                        "events" in data[key][s] and "generated-by-split" in data[key][s]["events"]]) > 0]
        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def die_by_merge(self, source_keys=None, return_keys=False):
        source_keys = self._data.keys() if source_keys is None else source_keys
        data = {key: self._data[key] for key in source_keys}
        keys = [key for key in data
                if
                len([s for s in data[key] if "events" in data[key][s] and "merged/died" in data[key][s]["events"]]) > 0]
        if return_keys:
            return keys
        else:
            return [(k, data[k]) for k in keys]

    def most_population(self, top=None, source_keys=None, return_keys=False):
        source_keys = self._data.keys() if source_keys is None else source_keys
        data = {key: self._data[key] for key in source_keys}

        keys = []
        for comm_id in data:
            population_timeline = [len(data[comm_id][s]["cur_members"]) for s in data[comm_id]
                                   if "cur_members" in data[comm_id][s]]
            largest_population = sum(population_timeline) / len(population_timeline) \
                if len(population_timeline) > 0 else 0
            keys.append((comm_id, largest_population))
        keys.sort(key=lambda e: e[1], reverse=True)  # 从大到小对平均成员规模排序
        if return_keys:
            return [t[0] for t in keys][:top] if top else [t[0] for t in keys]
        else:
            return [(t[0], t[1]) for t in keys][:top] if top else [(t[0], t[1]) for t in keys]

    def min_variance(self, top=None, source_keys=None, return_keys=False):
        source_keys = self._data.keys() if source_keys is None else source_keys
        data = {key: self._data[key] for key in source_keys}

        keys = []
        for comm_id in data:
            population_timeline = [len(data[comm_id][s]["cur_members"]) for s in data[comm_id]
                                   if "cur_members" in data[comm_id][s]]
            mean = sum(population_timeline) / len(population_timeline) \
                if len(population_timeline) > 0 else 0
            variance = sum([(v - mean) ** 2 for v in population_timeline]) / len(population_timeline)
            keys.append((comm_id, variance))
        keys.sort(key=lambda e: e[1])  # 从小到大对方差排序
        if return_keys:
            return [t[0] for t in keys][:top] if top else [t[0] for t in keys]
        else:
            return [(t[0], t[1]) for t in keys][:top] if top else [(t[0], t[1]) for t in keys]

    # ==========================================================================
    # 数据储存
    # ==========================================================================

    def dump_spec_comm(self, comm_id, dump_json_path):
        with open(dump_json_path, "w", encoding="utf8") as wf:
            json.dump(self._data[comm_id], wf, indent=2)

    def dump_network_profile(self, out_dir=None, return_obj=False):
        """
        1. 生成全网络的社区数据量变化图
        2. 生成全网络的事件数量变化图
        数据格式如下：
        [
            {
                slice_id: 8,
                datetime_interval: "Starting 2012-03-09 08:05:12 ending 2012-03-18 04:30:58",
                value: 500
            },
            ......
        ]
        """
        total_slices = set(self.death_count.keys()) | set(self.birth_count.keys()) | set(self.merge_count.keys()) \
                       | set(self._comm_count)

        # 事件数量计数文件
        num_events_out_data = [{
            "slice_id": s_id,
            "datetime_interval": self._real_datetime_map[s_id].split(" ")[4]
            if s_id in self._real_datetime_map else "",
            "comm_count": self._comm_count.get(s_id, 0),
            "born_count": self.birth_count.get(s_id, 0),
            "die_count": self.death_count.get(s_id, 0),
            "merge_count": self.merge_count.get(s_id, 0),
            "split_count": self.split_count.get(s_id, 0)
        } for s_id in total_slices]

        if return_obj:
            return num_events_out_data
        else:
            out_dir = out_dir if out_dir.endswith("/") else out_dir + "/"
            num_events_out_path = out_dir + "network_profile.json"
            with open(num_events_out_path, "w", encoding="utf8") as wf:
                json.dump(num_events_out_data, wf, indent=2)


def main():
    selector = CommSelector(tiles_output_dir="../../../output/bot10_30_240/")

    # 选取生存时间较长的，且由分裂产生的社区
    long_live_born_from_split = selector.born_from_split(
        source_keys=selector.most_long_live_comms(top=1000, return_keys=True))

    # 最佳多样性
    most_variety = selector.most_merging(
        return_keys=True,
        top=5, source_keys=selector.most_long_live_comms(
            top=50, return_keys=True, source_keys=selector.most_splitting(top=100, return_keys=True)))

    for rank, comm_id in enumerate(most_variety):
        selector.dump_spec_comm(comm_id=comm_id, dump_json_path=f"./rank{rank}_{comm_id}_dump.json")


if __name__ == '__main__':
    main()
