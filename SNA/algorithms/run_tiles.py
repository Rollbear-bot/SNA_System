from SNA.algorithms.tiles.alg.TILES import TILES


# 运行TILES算法
def run_tiles(file_path, dump_dir):
    obs = 30
    ttl = 240

    tag = open(dump_dir + "tiles.tag", "w", encoding="utf8")
    tag.write("运行中")
    tag.close()

    try:
        model = TILES(obs=obs, ttl=ttl, filename=file_path, path=dump_dir)
        model.execute()

        tag = open(dump_dir + "tiles.tag", "w", encoding="utf8")
        tag.write("已完成")
        tag.close()

    except Exception as e:
        print(e)
        tag = open(dump_dir + "tiles.tag", "w", encoding="utf8")
        tag.write("出错")
        tag.close()
