# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/12/24 9:56
# FileName: main
# Project: RemoteFaceRecognitionService
# Author: JasonLai
# Email: jasonlaihj@163.com
import os

import cv2
import uvicorn
import face_recognition
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException

from plugin import logger

# ==================== FAST ====================
app = FastAPI(title="Stable version face recognition system")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_SCALE = 0.25
BOX_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 创建人脸库目录 Create a face database directory
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# 全局变量（单线程，无锁）Global variables (single-threaded, no locks)
known_face_encodings = []
known_face_names = []
cap = None  # 摄像头全局变量（单线程独占）Camera global variable (single-thread exclusive)


# ==================== 稳定的人脸库加载逻辑 Stable face database loading logic ====================
def load_face_database():
    """同步加载（原始逻辑，无异步/线程）"""
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        try:
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                logger.info(f"Warning: {filename} has no face, skipped")
                continue
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        except Exception as e:
            logger.info(f"Failed to process {filename}: {e}")
    logger.info(f"Face database loaded successfully:{len(known_face_names)} people")


# ==================== 稳定的摄像头初始化 Stable camera initialization ====================
def init_camera():
    """单线程初始化 Single-thread initialization"""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap.isOpened()


# ==================== WebSocket logic ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_frame(self, frame_bytes: bytes):
        for conn in self.active_connections:
            await conn.send_bytes(frame_bytes)


manager = ConnectionManager()


# ==================== API 接口（上传同步刷库，牺牲速度换稳定） ====================
@app.get("/")
async def get_client():
    return FileResponse("static/index.html")


@app.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
    """上传同步刷库（原始单线程，无后台任务） Upload and synchronize database brushing (original single-threaded, no background tasks) """
    # 1. 格式校验 Format verification
    allowed = {"jpg", "jpeg", "png", "bmp"}
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Only supports {allowed} format")

    # 2. 保存文件 save face pictures
    file_path = os.path.join(KNOWN_FACES_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"文件保存：{file_path}")

        # 3. 同步刷库（无异步）Synchronous database brushing (no asynchronous)
        load_face_database()

        # 4. 响应（会慢一点，但稳定） Response (it will be a bit slow, but stable)
        return {
            "code": 200,
            "msg": f"Upload successful! The face database has been updated (currently {len(known_face_names)} people)",
            "count": len(known_face_names)
        }
    except Exception as e:
        raise HTTPException(500, f"Save failed: {str(e)}")


@app.websocket("/ws/face-stream")
async def websocket_endpoint(websocket: WebSocket):
    """Recognite logic"""
    await manager.connect(websocket)
    # 初始化摄像头 Initialize the camera
    if not init_camera():
        await websocket.close(reason="Camera failure")
        return

    try:
        while True:
            # 原始帧读取（单线程） Original frame reading (single-threaded)
            ret, frame = cap.read()
            if not ret:
                continue

            # 识别逻辑（无线程/异步） Identify logic (no threads/asynchronous)
            # Image scaling: Use `cv2.resize` to scale down the original frame (`frame`)
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

            # Color space conversion: Convert the BGR format (default in OpenCV) of the resized frame to RGB format (
            # `rgb_small`) using `cv2.cvtColor` to meet the requirements of subsequent face recognition.
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Obtain face position information
            face_locs = face_recognition.face_locations(rgb_small)

            # Extract facial feature encoding, Used for subsequent face matching
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)

            # 匹配逻辑（无锁，单线程安全） Matching logic (lock-free, single-thread safe)
            for face_enc, face_loc in zip(face_encs, face_locs):
                matches = face_recognition.compare_faces(known_face_encodings, face_enc, TOLERANCE)
                name = "Unknown"
                if len(known_face_encodings) > 0:
                    dists = face_recognition.face_distance(known_face_encodings, face_enc)
                    best_idx = dists.argmin()
                    if matches[best_idx]:
                        name = known_face_names[best_idx]

                # 绘制逻辑 Drawing logic
                top, right, bottom, left = face_loc
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)
                cv2.rectangle(frame, (left, top), (right, bottom), BOX_COLOR, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), BOX_COLOR, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), FONT, 1.0, (255, 255, 255), 2)

            # 压缩推送 compressed push
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                await manager.send_frame(buffer.tobytes())

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.info(f"WS error: {e}")
        manager.disconnect(websocket)


# ==================== 启动（单进程、无重载） Start (single process, no reload) ====================
if __name__ == "__main__":
    # 初始化人脸库（启动时加载）Initialize the face database (loaded at startup)
    load_face_database()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # 关键：关闭热重载 Key: Turn off hot reloading
        workers=1  # 关键：单进程 Key: single process
    )
