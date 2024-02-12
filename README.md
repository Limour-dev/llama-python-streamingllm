## 整体效果
+ 视频演示

https://github.com/Limour-dev/llama-python-streamingllm/assets/93720049/6391cd79-8a90-40b2-95fe-cfe5b8ff3ac9

+ 聊天页面

![聊天](https://raw.githubusercontent.com/Limour-dev/llama-python-streamingllm/main/2024-02/chrome_rMYBToBhpg.webp)
+ 设置页面

![设置](https://raw.githubusercontent.com/Limour-dev/llama-python-streamingllm/main/2024-02/chrome_BfYDpFh9WA.webp)

## Huggingface Spaces
+ 先检查 Huggingface 是否正常：[status.huggingface.co](https://status.huggingface.co/)
+ [Limour](https://huggingface.co/Limour)/[llama-python-streamingllm](https://huggingface.co/spaces/Limour/llama-python-streamingllm)
+ 仅支持同时一个人用，用之前点 `Reset` 按钮恢复初始的 kv_cache，按 `Submit` 没反应，说明有人在用，等一段时间后再 `Reset`
+ 多于一个窗口使用会崩溃，需要到设置里 `Restart this Space` 才能恢复
+ 只能 Duplicate 后，设为私密来使用

## 本地安装
```powershel
conda create -n llamaCpp libcublas cuda-toolkit git -c nvidia -c conda-forge
conda activate llamaCpp
conda install python=3.10 gradio -c conda-forge
# 然后去 release 下载相应的包 https://github.com/Limour-dev/llama-cpp-python-cuBLAS-wheels/releases
pip install --force-reinstall llama_cpp_python-0.2.39+cu122-cp310-cp310-win_amd64.whl
git clone --depth=1 https://github.com/Limour-dev/llama-python-streamingllm.git
cd llama-python-streamingllm
mkdir cache
python .\gradio_streamingllm.py
```

## 核心思想
借助 llama.cpp 的 kv_cache_seq_rm 和 kv_cache_seq_shift 两个 api 实现对 kv_cache 的 token 级操作 
 
定义了 venv 开头的方法，用于对 kv_cache 的 token 进行标记，比如 51-100是RAG注入的内容 101-150是用户的输入 151-200是旁白之类的 
 
在此基础之上就可以动态将不再用到的 token 进行 remove 以节省 kv_cache 的空间。 
 
最后就是当 kv_cache 满了之后，先跳过指定要永久保留的内容，比如system，然后从开头开始，从 kv_cache 中动态移除不再用到的 token 
