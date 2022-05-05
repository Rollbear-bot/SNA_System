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


def get_some_comm_meta(tiles_dump_dir):
    selector = CommSelector(tiles_output_dir=tiles_dump_dir,
                            load_elt=False)
    most_merge = selector.most_merging(top=100, return_keys=True)
    most_popu_comms = selector.most_population(top=50, source_keys=most_merge, return_keys=True)
    min_variance = selector.min_variance(top=10, source_keys=most_popu_comms, return_keys=True)

    meta_data = []
    for comm in min_variance:
        slice_id_s = sorted(selector.data[comm])[0]
        slice_id_e = sorted(selector.data[comm])[-1]
        most_popu_slice = sorted(selector.data[comm],
                                 key=lambda s: len(selector.data[comm][s].get("cur_members", [])))[-1]
        comm_meta = {
            "c_id": comm,
            "datetime_s": selector.real_datetime_map[slice_id_s].split()[1]
            if slice_id_s in selector.real_datetime_map else "未知",
            "datetime_e": selector.real_datetime_map[slice_id_e].split()[4]
            if slice_id_e in selector.real_datetime_map else "未知",
            "num_slice": len(selector.data[comm]),
            "most_popu_str": f"该社区规模最大时具有 {len(selector.data[comm][most_popu_slice].get('cur_members', []))} 个节点，"
                             f"位于时间片[{most_popu_slice}] {selector.real_datetime_map[most_popu_slice].split()[1]} ~ {selector.real_datetime_map[most_popu_slice].split()[4]}."
        }
        meta_data.append(comm_meta)

    return meta_data


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
