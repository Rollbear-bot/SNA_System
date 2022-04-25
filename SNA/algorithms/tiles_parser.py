from SNA.algorithms.parse.comm_select_v211229 import CommSelector
from SNA.algorithms.parse.comm_trace_v211229 import CommTrace


def get_network_meta(tiles_dump_dir):
    selector = CommSelector(tiles_dump_dir)
    return {
        "birth_count": [int(e[1]) for e in sorted(selector.birth_count.items())],
        "death_count": [int(e[1]) for e in sorted(selector.death_count.items())],
        "split_count": [int(e[1]) for e in sorted(selector.split_count.items())],
        "merge_count": [int(e[1]) for e in sorted(selector.merge_count.items())],
        "comm_count": [int(e[1]) for e in sorted(selector.comm_count.items())],
        "x_axis": sorted(selector.birth_count.keys())
    }
