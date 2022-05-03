
import os


def get_file_size(path):
    file_stats = os.stat(path)
    # print(file_stats)
    # print(f'File Size in Bytes is {file_stats.st_size}')
    # print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    return f"{file_stats.st_size / (1024 * 1024):.4f} MB"
