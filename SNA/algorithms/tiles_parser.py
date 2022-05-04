from SNA.algorithms.parse.comm_select_v211229 import CommSelector
from SNA.algorithms.parse.comm_trace_v211229 import CommTrace


def _map_real_date(slice_id, selector):
    return selector.real_datetime_map[slice_id].split()[1] if slice_id in selector.real_datetime_map else ""


def get_network_meta(tiles_dump_dir):
    selector = CommSelector(tiles_dump_dir)
    return {
        "birth_count": [int(e[1]) for e in sorted(selector.birth_count.items())],
        "death_count": [int(e[1]) for e in sorted(selector.death_count.items())],
        "split_count": [int(e[1]) for e in sorted(selector.split_count.items())],
        "merge_count": [int(e[1]) for e in sorted(selector.merge_count.items())],
        "comm_count": [int(e[1]) for e in sorted(selector.comm_count.items())],
        "x_axis": [_map_real_date(s_id, selector) for s_id in sorted(selector.birth_count.keys())]
    }


def get_some_comm_id(tiles_dump_dir):
    selector = CommSelector(tiles_output_dir=tiles_dump_dir,
                            load_elt=False)
    most_merge = selector.most_merging(top=100, return_keys=True)
    most_popu_comms = selector.most_population(top=50, source_keys=most_merge, return_keys=True)
    min_variance = selector.min_variance(top=10, source_keys=most_popu_comms, return_keys=True)
    return min_variance


def get_slices_and_data(tiles_dump_dir, c_id):
    tracer = CommTrace(comm_id=str(c_id), tiles_output_dir=tiles_dump_dir)
    slices = {}
    for k, v in tracer.data.items():
        if "edgelist" in v:
            slices[k] = {
                "members": v["cur_members"],
                "edgelist": v["edgelist"],
                "slice_id": k,
                "datetime_s": v["datetime"].split()[1],
                "datetime_e": v["datetime"].split()[4],
                "events": list(v["events"].keys()) if "events" in v else [],
            }
    slices = [e[1] for e in sorted(slices.items(), key=lambda e: e[0])]
    assert len(slices) > 0
    return slices


if __name__ == '__main__':
    tiles_output_dir = "../alg_result/20/"

    get_slices_and_data(tiles_output_dir, "535")
