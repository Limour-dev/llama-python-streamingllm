import hashlib
import json


# ========== 哈希函数 ==========
def x_hash(x: str):
    return hashlib.sha1(x.encode('utf-8')).hexdigest()


# ========== 读取配置文件 ==========
with open('rp_config.json', encoding='utf-8') as f:
    tmp = f.read()
with open('rp_sample_config.json', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['setting_cache_path']['value'] += x_hash(tmp)
cfg.update(json.loads(tmp))
