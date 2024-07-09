#!/bin/bash

# 脚本名称: run_test.py.sh

# 定义要运行的Python脚本名称
PYTHON_SCRIPT="lora_whisper_homework_script.py"

# 使用nohup运行Python脚本，并将输出重定向到nohup.out文件
nohup python $PYTHON_SCRIPT --mode test &

# 输出nohup命令的进程ID
echo "Python脚本已使用nohup启动，进程ID: $!"
