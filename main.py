import io
import os

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from captcha_cracker import CaptchaModel

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, "models", "weights.h5")

captcha_model = CaptchaModel()


@app.post("/", summary="CAPTCHA IMAGE")
async def upload(file: UploadFile = File(...)):
    read_image = Image.open(io.BytesIO(await file.read()))

    if read_image.mode == "RGBA":
        background = Image.new("RGB", read_image.size, (255, 255, 255))
        background.paste(read_image, (0, 0), read_image)
        image = background
    else:
        image = read_image.convert("RGB")

    img_byte_arr = io.BytesIO()

    image.save(img_byte_arr, format="PNG")

    img_byte_arr.seek(0)

    prediction = captcha_model.predict_from_bytes(img_byte_arr.getvalue())

    return {"content_type": file.content_type, "prediction": prediction}
