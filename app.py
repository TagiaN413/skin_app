from fastapi import FastAPI
from fastapi import UploadFile, File
import main
import numpy as np
import cv2
from fastapi.responses import FileResponse
from enum import Enum

app = FastAPI()
face_app = main.face_app()
class PredictName(str,Enum):
    Acne = "Acne"
    Wrinkles = "Wrinkles"
    SkinColor = 'SkinColor'
    
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

@app.post('/api/{predict_object}')
async def predict_image(predict_object:PredictName, file: UploadFile = File(...)):
    random = await file.read()
    image = np.fromstring(random,np.uint8)
    face = face_app.face_detect(image)
    mask = face_app.make_mask(face)
    if predict_object.value == 'Acne':
        acne = face_app.acne_detect(mask)
        cv2.imwrite('response.png', acne)
        return FileResponse('response.png')
    if predict_object.value == 'Wrinkles':
        wrinkles = face_app.wrinkle_detect(mask)
        cv2.imwrite('response.png', wrinkles)
        return FileResponse('response.png')    
    if predict_object.value == 'SkinColor':
        skinRGB = face_app.skin_color(mask)
        res = {
            "RGB" : skinRGB
        }
        return res
    

'''
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item 
'''
