import os


def init(cfg):
    if os.path.exists(cfg['setting_cache_path'].value):
        # ========== 加载角色卡-缓存 ==========
        tmp = cfg['model'].load_session(cfg['setting_cache_path'].value)
        print(f"load cache from {cfg['setting_cache_path'].value} {tmp}")
        tmp = cfg['chat_template']('system',
                                   cfg['text_format'](cfg['role_char_d'].value,
                                                      char=cfg['role_char'].value,
                                                      user=cfg['role_usr'].value))
        cfg['setting_n_keep'].value = len(tmp)
        tmp = cfg['chat_template'](cfg['role_char'].value,
                                   cfg['text_format'](cfg['role_chat_style'].value,
                                                      char=cfg['role_char'].value,
                                                      user=cfg['role_usr'].value))
        cfg['setting_n_keep'].value += len(tmp)
        # ========== 加载角色卡-第一条消息 ==========
        cfg['chatbot'] = []
        for one in cfg["role_char_first"]:
            one['name'] = cfg['text_format'](one['name'],
                                             char=cfg['role_char'].value,
                                             user=cfg['role_usr'].value)
            one['value'] = cfg['text_format'](one['value'],
                                              char=cfg['role_char'].value,
                                              user=cfg['role_usr'].value)
            if one['name'] == cfg['role_char'].value:
                cfg['chatbot'].append((None, cfg['chat_display_format'](one['value'])))
            print(one)
    else:
        # ========== 加载角色卡-角色描述 ==========
        tmp = cfg['chat_template']('system',
                                   cfg['text_format'](cfg['role_char_d'].value,
                                                      char=cfg['role_char'].value,
                                                      user=cfg['role_usr'].value))
        cfg['setting_n_keep'].value = cfg['model'].eval_t(tmp)  # 此内容永久存在

        # ========== 加载角色卡-回复示例 ==========
        tmp = cfg['chat_template'](cfg['role_char'].value,
                                   cfg['text_format'](cfg['role_chat_style'].value,
                                                      char=cfg['role_char'].value,
                                                      user=cfg['role_usr'].value))
        cfg['setting_n_keep'].value = cfg['model'].eval_t(tmp)  # 此内容永久存在

        # ========== 加载角色卡-第一条消息 ==========
        cfg['chatbot'] = []
        for one in cfg["role_char_first"]:
            one['name'] = cfg['text_format'](one['name'],
                                             char=cfg['role_char'].value,
                                             user=cfg['role_usr'].value)
            one['value'] = cfg['text_format'](one['value'],
                                              char=cfg['role_char'].value,
                                              user=cfg['role_usr'].value)
            if one['name'] == cfg['role_char'].value:
                cfg['chatbot'].append((None, cfg['chat_display_format'](one['value'])))
            print(one)
            tmp = cfg['chat_template'](one['name'], one['value'])
            cfg['model'].eval_t(tmp)  # 此内容随上下文增加将被丢弃

        # ========== 保存角色卡-缓存 ==========
        with open(cfg['setting_cache_path'].value, 'wb') as f:
            pass
        tmp = cfg['model'].save_session(cfg['setting_cache_path'].value)
        print(f'save cache {tmp}')
