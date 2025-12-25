# RemoteFaceRecognitionService
A real-time face recognition system was built using the FastAPI framework and OpenCV.
# 项目名称
这是一个实时人脸识别功能的Python项目，用于室内监控场景或者个人登录验证场景等等。

## 功能介绍
- 功能1：实现模板人脸库上传功能
- 功能2：支持人脸实时识别
- 功能3：实现多终端介入

## 环境要求
- Python 3.9.11
- 依赖库：fastapi、python-openCV、face-recognition、dlib、uvicorn、numpy、websockets

## 使用方法
```bash
# 安装依赖
pip install -r requirements.txt
# 运行主程序
python main.py

## 注意
在安装及使用face-recognition之前需要先安装c++编辑器 visual studio，接着安装dlib，face-recognition高度依赖dlib
