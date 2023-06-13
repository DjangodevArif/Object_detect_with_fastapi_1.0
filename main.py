
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO
import uuid
from PIL import Image
import shutil
import os
import io
import tempfile
import cv2
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect_video")
async def read_item(file: UploadFile):
    file_content = await file.read()
    unique = str(uuid.uuid4()).split("-")[0]

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    output_file_path = temp_file.name
    save_dir_for_video = os.getcwd() + "/raw/"+unique+"/"
    if not os.path.exists(save_dir_for_video):
        os.makedirs(save_dir_for_video)
    video_path = save_dir_for_video+"video.mp4"

    await save_video_file(file_content, output_file_path, save_dir_for_video)
    model = YOLO('yolov8n.pt')
    file_name = "result_" + unique
    results = model(video_path, save=True,
                    project="detect", name=file_name)
    detect_image = "detect/"+file_name+"/video.mp4"

    # Removing directory
    shutil.rmtree(video_path)

    return FileResponse(detect_image)


@app.post("/detect_image")
async def read_item(file: UploadFile):
    file_content = await file.read()

    unique = str(uuid.uuid4()).split("-")[0]
    nparr = np.frombuffer(file_content, dtype=np.uint8)
    cv2_image = cv2.imdecode(nparr, 1)
    model = YOLO('yolov8n.pt')
    file_name = "result_" + unique
    results = model(cv2_image, save=True,
                    project="detect", name=file_name)
    detect_image = "detect/"+file_name+"/image0.jpg"

    return FileResponse(detect_image)


async def save_video_file(bytes, output_file_path, save_dir_for_video):
    with open(output_file_path, 'wb') as f:
        f.write(bytes)

    # Replace with your desired destination path
    shutil.move(output_file_path, save_dir_for_video+"video.mp4")


@app.get("/file_stream")
async def file_stream():
    def iterfile():  #
        with open("sample.gif", mode="rb") as file_like:  #
            yield from file_like  #

    return StreamingResponse(iterfile(), media_type="video/gif")
