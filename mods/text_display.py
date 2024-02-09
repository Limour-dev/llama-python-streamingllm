import re


#  ========== 适配 SillyTavern 的模版 ==========
def text_format(text: str, _env=None, **env):
    if _env is not None:
        for k, v in _env.items():
            text = text.replace(r'{{' + k + r'}}', v)
    for k, v in env.items():
        text = text.replace(r'{{' + k + r'}}', v)
    return text


#  ========== 给引号加粗 ==========
reg_q = re.compile(r'“(.+?)”')


def chat_display_format(text: str):
    return reg_q.sub(r' **\g<0>** ', text)


def init(cfg):
    cfg['text_format'] = text_format
    cfg['chat_display_format'] = chat_display_format
