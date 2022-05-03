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
