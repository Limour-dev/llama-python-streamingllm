import numpy as np


def init(cfg):
    chat_template = cfg['chat_template']
    model = cfg['model']
    s_info = cfg['s_info']
    lock = cfg['session_lock']

    # ========== 预处理 key、desc ==========
    def str_tokenize(s):
        s = model.tokenize((chat_template.nl + s).encode('utf-8'), add_bos=False, special=False)
        if s[0] in chat_template.onenl:
            return s[1:]
        else:
            return s

    text_format = cfg['text_format']
    for x in cfg['btn_status_bar_list']:
        x['key'] = text_format(x['key'],
                               char=cfg['role_char'].value,
                               user=cfg['role_usr'].value)
        x['key_t'] = str_tokenize(x['key'])
        x['desc'] = text_format(x['desc'],
                                char=cfg['role_char'].value,
                                user=cfg['role_usr'].value)
        if x['desc']:
            x['desc_t'] = str_tokenize(x['desc'])

    # ========== 预处理 构造函数 mask ==========
    def btn_status_bar_fn_mask():
        _shape1d = model.scores.shape[-1]
        mask = np.full((_shape1d,), -np.inf, dtype=np.single)
        return mask

    # ========== 预处理 构造函数 数字 ==========
    def btn_status_bar_fn_int(unit: str):
        t_int = str_tokenize('0123456789')
        assert len(t_int) == 10
        fn_int_mask = btn_status_bar_fn_mask()
        fn_int_mask[chat_template.eos] = 0
        fn_int_mask[t_int] = 0
        if unit:
            unit_t = str_tokenize(unit)
            fn_int_mask[unit_t[0]] = 0

        def logits_processor(_input_ids, logits):
            return logits + fn_int_mask

        def inner(eval_t, sample_t):
            retn = []
            while True:
                token = sample_t(logits_processor)
                # ========== 不是数字就结束 ==========
                if token in chat_template.eos:
                    break
                if unit and token == unit_t[0]:
                    break
                # ========== 是数字就继续 ==========
                retn.append(token)
                eval_t([token])

            if unit:
                eval_t(unit_t)  # 添加单位
                retn.extend(unit_t)

            return model.str_detokenize(retn)

        return inner

    # ========== 预处理 构造函数 集合 ==========
    def btn_status_bar_fn_set(value):
        value_t = {_x[0][0]: _x for _x in ((str_tokenize(_y), _y) for _y in value)}
        fn_set_mask = btn_status_bar_fn_mask()
        fn_set_mask[list(value_t.keys())] = 0

        def logits_processor(_input_ids, logits):
            return logits + fn_set_mask

        def inner(eval_t, sample_t):
            token = sample_t(logits_processor)
            eval_t(value_t[token][0])
            return value_t[token][1]

        return inner

    # ========== 预处理 构造函数 字符串 ==========
    def btn_status_bar_fn_str():
        def inner(eval_t, sample_t):
            retn = []
            tmp = ''
            while True:
                token = sample_t(None)
                if token in chat_template.eos:
                    break
                retn.append(token)
                tmp = model.str_detokenize(retn)
                if tmp.endswith('\n') or tmp.endswith('\r'):
                    break
                # ========== 继续 ==========
                eval_t([token])
            return tmp.strip()

        return inner

    # ========== 预处理 value ==========
    for x in cfg['btn_status_bar_list']:
        for y in x['combine']:
            if y['prefix']:
                y['prefix_t'] = str_tokenize(y['prefix'])

            if y['type'] == 'int':
                y['fn'] = btn_status_bar_fn_int(y['unit'])
            elif y['type'] == 'set':
                y['fn'] = btn_status_bar_fn_set(y['value'])
            elif y['type'] == 'str':
                y['fn'] = btn_status_bar_fn_str()
            else:
                pass

    # ========== 添加分隔标记 ==========
    for i, x in enumerate(cfg['btn_status_bar_list']):
        if i == 0:  # 跳过第一个
            continue
        x['key_t'] = chat_template.im_end_nl[-1:] + x['key_t']

    del x  # 避免干扰
    del y

    # print(cfg['btn_status_bar_list'])

    # ========== 输出状态栏 ==========
    def btn_status_bar(_n_keep, _n_discard,
                       _temperature, _repeat_penalty, _frequency_penalty,
                       _presence_penalty, _repeat_last_n, _top_k,
                       _top_p, _min_p, _typical_p,
                       _tfs_z, _mirostat_mode, _mirostat_eta,
                       _mirostat_tau, _usr, _char,
                       _rag, _max_tokens):
        with lock:
            if not cfg['session_active']:
                raise RuntimeError
            if cfg['btn_stop_status']:
                yield [], model.venv_info
                return

            # ========== 临时的eval和sample ==========
            def eval_t(tokens):
                return model.eval_t(
                    tokens=tokens,
                    n_keep=_n_keep,
                    n_discard=_n_discard,
                    im_start=chat_template.im_start_token
                )

            def sample_t(logits_processor):
                return model.sample_t(
                    top_k=_top_k,
                    top_p=_top_p,
                    min_p=_min_p,
                    typical_p=_typical_p,
                    temp=_temperature,
                    repeat_penalty=_repeat_penalty,
                    repeat_last_n=_repeat_last_n,
                    frequency_penalty=_frequency_penalty,
                    presence_penalty=_presence_penalty,
                    tfs_z=_tfs_z,
                    mirostat_mode=_mirostat_mode,
                    mirostat_tau=_mirostat_tau,
                    mirostat_eta=_mirostat_eta,
                    logits_processor=logits_processor
                )

            # ========== 初始化输出模版 ==========
            model.venv_create('status')  # 创建隔离环境
            eval_t(chat_template('状态'))  # 开始标记
            # ========== 流式输出 ==========
            df = []  # 清空
            for _x in cfg['btn_status_bar_list']:
                # ========== 属性 ==========
                df.append([_x['key'], ''])
                eval_t(_x['key_t'])
                if _x['desc']:
                    eval_t(_x['desc_t'])
                yield df, model.venv_info
                # ========== 值 ==========
                for _y in _x['combine']:
                    if _y['prefix']:
                        if df[-1][-1]:
                            df[-1][-1] += _y['prefix']
                        else:
                            df[-1][-1] += _y['prefix'].lstrip(':')
                        eval_t(_y['prefix_t'])
                    df[-1][-1] += _y['fn'](eval_t, sample_t)
                    yield df, model.venv_info
            eval_t(chat_template.im_end_nl)  # 结束标记
            # ========== 清理上一次生成的状态栏 ==========
            model.venv_remove('status', keep_last=1)
            yield df, model.venv_info

    cfg['btn_status_bar_fn'] = {
        'fn': btn_status_bar,
        'inputs': cfg['setting'],
        'outputs': [cfg['status_bar'], s_info]
    }
    cfg['btn_status_bar_fn'].update(cfg['btn_concurrency'])

    cfg['btn_status_bar'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_status_bar_fn']
    ).success(
        **cfg['btn_finish']
    )
