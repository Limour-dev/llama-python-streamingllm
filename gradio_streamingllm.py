import gradio as gr
import threading
from llama_cpp_python_streamingllm import StreamingLLM
from mods.read_cfg import cfg

from mods.text_display import init as text_display_init

from mods.btn_rag import init as btn_rag_init

# ========== 按钮中用到的共同的函数 ==========
from mods.btn_com import init as btn_com_init

# ========== 输出一段回答 ==========
from mods.btn_submit import init as btn_submit_init

# ========== 输出一段旁白 ==========
from mods.btn_vo import init as btn_vo_init

# ========== 重新输出一段回答 ==========
from mods.btn_retry import init as btn_retry_init

# ========== 给用户提供默认回复的建议 ==========
from mods.btn_suggest import init as btn_suggest_init

# ========== 重置按钮 ==========
from mods.btn_reset import init as btn_reset_init

# ========== 聊天的模版 默认 chatml ==========
from chat_template import ChatTemplate

# ========== 加载角色卡-缓存 ==========
from mods.load_cache import init as load_cache_init

#  ========== 全局锁，确保只能进行一个会话 ==========
cfg['session_lock'] = threading.Lock()
cfg['session_active'] = False

#  ========== 温度、采样之类的设置 ==========
with gr.Blocks() as setting:
    with gr.Row():
        cfg['setting_path'] = gr.Textbox(label="模型路径", max_lines=1, scale=2, **cfg['setting_path'])
        cfg['setting_cache_path'] = gr.Textbox(label="缓存路径", max_lines=1, scale=2, **cfg['setting_cache_path'])
        cfg['setting_seed'] = gr.Number(label="随机种子", scale=1, **cfg['setting_seed'])
        cfg['setting_n_gpu_layers'] = gr.Number(label="n_gpu_layers", scale=1, **cfg['setting_n_gpu_layers'])
    with gr.Row():
        cfg['setting_ctx'] = gr.Number(label="上下文大小（Tokens）", **cfg['setting_ctx'])
        cfg['setting_max_tokens'] = gr.Number(label="最大响应长度（Tokens）", interactive=True,
                                              **cfg['setting_max_tokens'])
        cfg['setting_n_keep'] = gr.Number(value=10, label="n_keep", interactive=False)
        cfg['setting_n_discard'] = gr.Number(label="n_discard", interactive=True, **cfg['setting_n_discard'])
    with gr.Row():
        cfg['setting_temperature'] = gr.Number(label="温度", interactive=True, **cfg['setting_temperature'])
        cfg['setting_repeat_penalty'] = gr.Number(label="重复惩罚", interactive=True, **cfg['setting_repeat_penalty'])
        cfg['setting_frequency_penalty'] = gr.Number(label="频率惩罚", interactive=True,
                                                     **cfg['setting_frequency_penalty'])
        cfg['setting_presence_penalty'] = gr.Number(label="存在惩罚", interactive=True,
                                                    **cfg['setting_presence_penalty'])
        cfg['setting_repeat_last_n'] = gr.Number(label="惩罚范围", interactive=True, **cfg['setting_repeat_last_n'])
    with gr.Row():
        cfg['setting_top_k'] = gr.Number(label="Top-K", interactive=True, **cfg['setting_top_k'])
        cfg['setting_top_p'] = gr.Number(label="Top P", interactive=True, **cfg['setting_top_p'])
        cfg['setting_min_p'] = gr.Number(label="Min P", interactive=True, **cfg['setting_min_p'])
        cfg['setting_typical_p'] = gr.Number(label="Typical", interactive=True, **cfg['setting_typical_p'])
        cfg['setting_tfs_z'] = gr.Number(label="TFS", interactive=True, **cfg['setting_tfs_z'])
    with gr.Row():
        cfg['setting_mirostat_mode'] = gr.Number(label="Mirostat 模式", **cfg['setting_mirostat_mode'])
        cfg['setting_mirostat_eta'] = gr.Number(label="Mirostat 学习率", interactive=True,
                                                **cfg['setting_mirostat_eta'])
        cfg['setting_mirostat_tau'] = gr.Number(label="Mirostat 目标熵", interactive=True,
                                                **cfg['setting_mirostat_tau'])

    #  ========== 加载模型 ==========
    cfg['model'] = StreamingLLM(model_path=cfg['setting_path'].value,
                                seed=cfg['setting_seed'].value,
                                n_gpu_layers=cfg['setting_n_gpu_layers'].value,
                                n_ctx=cfg['setting_ctx'].value)
    cfg['chat_template'] = ChatTemplate(cfg['model'])
    cfg['setting_ctx'].value = cfg['model'].n_ctx()

# ========== 展示角色卡 ==========
with gr.Blocks() as role:
    with gr.Row():
        cfg['role_usr'] = gr.Textbox(label="用户名称", max_lines=1, interactive=False, **cfg['role_usr'])
        cfg['role_char'] = gr.Textbox(label="角色名称", max_lines=1, interactive=False, **cfg['role_char'])

    cfg['role_char_d'] = gr.Textbox(lines=10, label="故事描述", **cfg['role_char_d'])
    cfg['role_chat_style'] = gr.Textbox(lines=10, label="回复示例", **cfg['role_chat_style'])

    # ========== 加载角色卡-缓存 ==========
    text_display_init(cfg)
    load_cache_init(cfg)

# ========== 聊天页面 ==========
with gr.Blocks() as chatting:
    with gr.Row(equal_height=True):
        cfg['chatbot'] = gr.Chatbot(height='60vh', scale=2, value=cfg['chatbot'],
                                    avatar_images=(r'assets/user.png', r'assets/chatbot.webp'))
        with gr.Column(scale=1, elem_id="area"):
            cfg['rag'] = gr.Textbox(label='RAG', show_copy_button=True, elem_id="RAG-area")
            cfg['vo'] = gr.Textbox(label='VO', show_copy_button=True, elem_id="VO-area")
            cfg['s_info'] = gr.Textbox(value=cfg['model'].venv_info, max_lines=1, label='info', interactive=False)
    cfg['msg'] = gr.Textbox(label='Prompt', lines=2, max_lines=2, elem_id='prompt', autofocus=True, **cfg['msg'])

    cfg['gr'] = gr
    btn_com_init(cfg)

    btn_rag_init(cfg)

    btn_submit_init(cfg)

    btn_vo_init(cfg)

    btn_suggest_init(cfg)

    btn_retry_init(cfg)

    # ========== 用于调试 ==========
    btn_reset_init(cfg)

#  ========== 让聊天界面的文本框等高 ==========
custom_css = r'''
#area > div {
    height: 100%;
}
#RAG-area {
    flex-grow: 1;
}
#RAG-area > label {
    height: 100%;
    display: flex;
    flex-direction: column;
}
#RAG-area > label > textarea {
    flex-grow: 1;
    max-height: 20vh;
}
#VO-area {
    flex-grow: 1;
}
#VO-area > label {
    height: 100%;
    display: flex;
    flex-direction: column;
}
#VO-area > label > textarea {
    flex-grow: 1;
    max-height: 20vh;
}
#prompt > label > textarea {
    max-height: 63px;
}
'''

# ========== 开始运行 ==========
demo = gr.TabbedInterface([chatting, setting, role],
                          ["聊天", "设置", '角色'],
                          css=custom_css)
gr.close_all()
demo.queue(api_open=False, max_size=1).launch(share=False, show_error=True, show_api=False)
