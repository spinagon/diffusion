import base64
import requests
import numpy as np
from io import BytesIO
import imageio
import datetime
from pathlib import Path

url = "http://192.168.0.100:7860/"


def txt2img(prompt, neg=""):
    r = requests.post(
        url + "sdapi/v1/txt2img",
        json={"prompt": prompt, "negative_prompt": neg, "steps": 30},
    )
    path = (
        "./sd/"
        + datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
        + ".png"
    )
    Path(path).write_bytes(base64.decodebytes(r.json()["images"][0].encode()))
    return path


def img2img(
    img,
    prompt="",
    neg="",
    denoise=0.6,
    seed=-1,
    sampler="LMS Karras",
    sd_model=None,
    mask=None,
    width=None,
    height=None,
    steps=30,
    scale=7,
):
    if isinstance(img, str):
        image = Path(img).read_bytes()
        h, w, *_ = imageio.imread(img).shape
    if isinstance(img, np.ndarray):
        h, w, *_ = img.shape
        img_8 = (img * 255).astype(np.uint8)
        data = BytesIO()
        imageio.imwrite(data, img_8, format="png")
        data.seek(0)
        image = data.read()

    if not width:
        width = round((np.sqrt((512 * 512) / (w * h)) * w) / 64) * 64
    if not height:
        height = round((np.sqrt((512 * 512) / (w * h)) * h) / 64) * 64

    r = requests.post(
        url + "sdapi/v1/img2img",
        json={
            "prompt": prompt,
            "steps": steps,
            "init_images": [base64.encodebytes(image).decode()],
            "denoising_strength": denoise,
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_index": sampler,
            "mask": mask,
            "negative_prompt": neg,
            "scale": scale,
        },
    )
    path = (
        "./sd/"
        + datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
        + ".png"
    )
    Path(path).write_bytes(base64.decodebytes(r.json()["images"][0].encode()))
    return path
