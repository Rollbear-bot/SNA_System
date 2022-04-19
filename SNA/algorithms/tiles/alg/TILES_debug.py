"""
用于TILES debug和运行结果review
接收带有tweet id字段的edgelist流
"""

"""
    Created on 11/feb/2015
    @author: Giulio Rossetti
"""
import datetime
import gzip
import os
import sys
import time

import networkx as nx
from future.utils import iteritems

if sys.version_info > (2, 7):
    from io import StringIO
    from queue import PriorityQueue
else:
    # 对Python2的支持
    from cStringIO import StringIO
    from Queue import PriorityQueue

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


class TILES_DEBUG(object):
    """
        TILES
        Algorithm for evolutionary community discovery
    """

    def __init__(self, filename=None, g=None, ttl=float('inf'), obs=7, path="", start=None, end=None,
                 tracing_nodes=None):
        """
            Constructor
            :param g: networkx graph
            :param ttl: edge time to live (days)
            :param obs: observation window (days)
            :param path: Path where generate the results and find the edge file
            :param start: starting date
            :param end: ending date
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
        self.communities = {}

        self.tracing_nodes = tracing_nodes
        self.tracing_nodes_log = ""

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

        # 先读取edgelist的第一行，初始化成员变量
        with open(self.filename, 'r') as edgelist_file:
            first_line = edgelist_file.readline()

        # 从时间戳获得datetime
        # 负数时间戳分开处理
        if float(first_line.split("\t")[2]) > 0:
            actual_time = datetime.datetime.fromtimestamp(float(first_line.split("\t")[2]))
        else:
            actual_time = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(first_line.split("\t")[2]))

        last_break = actual_time
        edgelist_file.close()

        count = 0

        #################################################
        #                   Main Cycle                  #
        #################################################

        # 从edgelist的第二行开始，进入主循环，每一次迭代即为处理一条边
        edgelist_file = open(self.filename)
        for line in edgelist_file:
            line = line.split("\t")
            self.added += 1
            e = {}
            u = int(line[0])
            v = int(line[1])
            # 当前边的datetime时间，负数时间戳分开处理
            if float(line[2]) > 0:
                dt = datetime.datetime.fromtimestamp(float(line[2]))
            else:
                dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(line[2]))

            e['weight'] = 1
            e["u"] = line[0]
            e["v"] = line[1]

            # 此处埋点（时间片编号）
            if self.actual_slice in [113, 114, 115, 116]:
                print("special slice!", self.actual_slice)

            # 此处埋点
            if e["u"] in self.tracing_nodes and e["v"] in self.tracing_nodes:
                self.tracing_nodes_log += f"tweet id: {line[3]}, edge {e['u']} -> {e['v']}, when: {str(dt)}\n"
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
                # 新边加入优先级队列，它的权值为1
                qr.put((dt, (int(e['u']), int(e['v']), int(e['weight']))))
                self.remove(dt, qr)  # 移除当前时间过期的边

            # 对两个端点u与v：如果是新出现的端点，则在数据结构中注册
            if not self.g.has_node(u):
                self.g.add_node(u)
                self.g.node[u]['c_coms'] = {}  # central
            if not self.g.has_node(v):
                self.g.add_node(v)
                self.g.node[v]['c_coms'] = {}

            # 如果当前边已经在图中存在，则它的权值+1，否则在数据结构中注册这条边，并将权值置为1
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
                        if w > 1:
                            self.g.adj[u][v]["weight"] = w - 1
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
                                        # todo::社区仅加入核心点？
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

            # todo::社区标签传播（核心点的传播）
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
                    # todo::边缘点不用加入社区？

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
        for k, idk in iteritems(nodes_to_coms):
            write_count += 1
            if write_count % 50000 == 0:
                out_file_coms.write(com_string.getvalue())
                out_file_coms.flush()  # 刷新文件缓冲区，将缓冲区的内容提前写入文件
                com_string = StringIO()
                write_count = 0
            com_string.write(u"%d\t%s\n" % (idk, str(list(k))))

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
            g_string.write(u"%d\t%s\t%d\n" % (e[0], e[1], self.g.adj[e[0]][e[1]]['weight']))

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


if __name__ == '__main__':
    filename = "D:\\py_backups\\SocialNetwork\\edgelists\\merged_sorted\\bot10_tweet_id.edgelist"
    # filename = "./edgelists/merged_sorted/bot10_tweet_id.edgelist"
    dump_dir = "../../output\\native_tiles_on_twitter/"
    # dump_dir = "./tiles_output/bot10_30_240_review/"

    # filename = sys.argv[1]
    # dump_dir = sys.argv[2]

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    obs = 30
    ttl = 240

    # obs = int(sys.argv[3])
    # ttl = sys.argv[4]

    if ttl == "inf":
        ttl = float("inf")
    else:
        ttl = float(ttl)

    model = TILES_DEBUG(obs=obs, ttl=ttl, filename=filename, path=dump_dir,
                        tracing_nodes=["1061678353", "17989291", "26257166", "181572333", "1917997100"])
    model.execute()

    print(model.tracing_nodes_log)
