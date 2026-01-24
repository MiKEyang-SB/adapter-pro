##沐曦服务器训练指南
先配置环境
conda create -n vla-adapter --clone base
直接把base的环境clone过来先
然后删除.toml里面的 torch toprchaudio torchvision
然后
pip install -e .
环境就配置完成了
还需要额外安装
numpy==1.26.4
positional-encodings==6.0.3