## 整体效果
+ 聊天页面

![聊天](https://raw.githubusercontent.com/Limour-dev/llama-python-streamingllm/main/2024-02/chrome_rMYBToBhpg.webp)
+ 设置页面

![设置](https://raw.githubusercontent.com/Limour-dev/llama-python-streamingllm/main/2024-02/chrome_BfYDpFh9WA.webp)

## 本地安装
```powershel
conda create -n llamaCpp libcublas cuda-toolkit git -c nvidia -c conda-forge
conda activate llamaCpp
conda install python=3.10 gradio -c conda-forge
# 然后去 release 下载相应的包 https://github.com/Limour-dev/llama-cpp-python-cuBLAS-wheels/releases
pip install --force-reinstall llama_cpp_python-0.2.39+cu122-cp310-cp310-win_amd64.whl
python .\gradio_streamingllm.py
```

## Huggingface Spaces
+ 先检查 Huggingface 是否正常：[status.huggingface.co](https://status.huggingface.co/)
+ [Limour](https://huggingface.co/Limour)/[llama-python-streamingllm](https://huggingface.co/spaces/Limour/llama-python-streamingllm)
+ 仅支持同时一个人用，用之前点 `Reset` 按钮恢复初始的 kv_cache，按 `Submit` 没反应，说明有人在用，等一段时间后再 `Reset`
+ 多于一个窗口使用会崩溃，需要到设置里 `Restart this Space` 才能恢复
+ 只能 Duplicate 后，设为私密来使用
