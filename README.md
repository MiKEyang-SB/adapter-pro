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
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

##查找setsid进程
ps -eo pid,ppid,sid,pgid,tty,cmd | grep torchrun
ps -u $USER -o pid,ppid,sid,pgid,tty,cmd | grep -E "torchrun|torch.distributed.run|finetune.py" | grep -v grep

kill -9 -PID

#杀评估进程
pkill -f "experiments/robot/libero/run_libero_eval.py"
pkill -u $USER -f "torchrun"