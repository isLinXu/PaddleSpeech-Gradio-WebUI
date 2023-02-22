#!/bin/bash



# 在线启动
gradio app.py

# 后台启动进程
#ps aux|grep root|grep app.py|awk '{print $2}'|xargs kill -9
#gradio app.py &