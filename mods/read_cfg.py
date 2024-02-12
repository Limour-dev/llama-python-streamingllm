import hashlib
import json
import os


def get_all_files_in_directory(directory, ext=''):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    return all_files


# ========== 哈希函数 ==========
def x_hash(x: str):
    return hashlib.sha1(x.encode('utf-8')).hexdigest()


# ========== 读取配置文件 ==========
with open('rp_config.json', encoding='utf-8') as f:
    tmp = f.read()
cfg = json.loads(tmp)
for path in get_all_files_in_directory('config', ext='.json'):
    with open(path, encoding='utf-8') as f:
        cfg.update(json.load(f))
cfg['setting_cache_path']['value'] += x_hash(tmp)

