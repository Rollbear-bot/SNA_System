# -*- coding: utf-8 -*-
# @Time: 2021/3/23 23:33
# @Author: Rollbear
# @Filename: explore.py

from comm_trace_v211229 import CommTrace
from comm_select_v211229 import CommSelector

tiles_output_dir = "../../alg_result/20/"


selector = CommSelector(tiles_output_dir=tiles_output_dir,
                        load_elt=False)
most_merge = selector.most_merging(top=100, return_keys=True)
most_popu_comms = selector.most_population(top=50, source_keys=most_merge, return_keys=True)
min_variance = selector.min_variance(top=5, source_keys=most_popu_comms, return_keys=False)


def trace_and_dump_profile(comm_id):
    tracer2 = CommTrace(comm_id=comm_id, tiles_output_dir=tiles_output_dir,
                        debug=True, parse_relative_sub_comm=False)
    tracer2.dump_community_profile(f"..\\..\\..\\output\\min_variance_comms\\community_{comm_id}_profile/")


for t in min_variance:
    trace_and_dump_profile(t[0])
