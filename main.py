import base64
import requests
import numpy as np
from io import BytesIO
import imageio
import datetime
from pathlib import Path

default_url = "http://127.0.0.1:7860/"


class Connection:
    def __init__(self, url):
        self.url = url

    def txt2img(self, prompt, neg="", **kwargs):
        payload = {"prompt": prompt, "negative_prompt": neg, "steps": 30}
        payload.update(kwargs)
        r = requests.post(
            self.url + "sdapi/v1/txt2img",
            json=payload,
        )
        path = self.prepare_path()
        if r.status_code != 200:
            print(r)
            print(r.text)
        paths = []
        for i, image in enumerate(r.json()["images"]):
            p = path + "_{}.png".format(i)
            paths.append(p)
            Path(p).write_bytes(base64.decodebytes(image.encode()))
        return paths

    def prepare_path(self):
        path = "./sd/" + datetime.datetime.now().isoformat().split(".")[0].replace(
            ":", "."
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def img2img(
        self,
        img,
        prompt="",
        neg="",
        denoise=0.6,
        seed=-1,
        sampler="DPM++ 2M Karras",
        sd_model=None,
        mask=None,
        width=None,
        height=None,
        steps=30,
        scale=7,
        inpaint_padding=64,
        **kwargs
    ):
        if isinstance(img, str):
            h, w, *_ = imageio.imread(img).shape
        if isinstance(img, np.ndarray):
            h, w, *_ = img.shape
        image = self.pack_image(img)
        if not width and not height:
            shorter = min(h, w)
            longer = max(h, w)
            longer = int(round(longer / shorter * 512 / 8) * 8)
            shorter = 512
            if w < h:
                width = shorter
                height = longer
            else:
                width = longer
                height = shorter

        payload = {
                "prompt": prompt,
                "steps": steps,
                "init_images": [image],
                "denoising_strength": denoise,
                "width": width,
                "height": height,
                "seed": seed,
                "sampler_index": sampler,
                "mask": mask,
                "negative_prompt": neg,
                "scale": scale,
                "inpaint_full_res_padding": inpaint_padding
            }
        payload.update(kwargs)
        r = requests.post(
            self.url + "sdapi/v1/img2img",
            json=payload
        )
        path = self.prepare_path()
        if r.status_code != 200:
            print(r)
            print(r.text)
            return
        paths = []
        for i, image in enumerate(r.json()["images"]):
            p = path + "_{}.png".format(i)
            paths.append(p)
            Path(p).write_bytes(base64.decodebytes(image.encode()))
        return paths

    def pack_image(self, img, format=None):
        if isinstance(img, str):
            image = Path(img).read_bytes()
            if format is None:
                format = Path(img).suffix.strip(".")
        if isinstance(img, np.ndarray):
            img_8 = (img * 255).astype(np.uint8)
            data = BytesIO()
            if format is None:
                format = "png"
            imageio.imwrite(data, img_8, format=format)
            data.seek(0)
            image = data.read()
        return base64.encodebytes(image).decode()

    def interrogate(self, img):
        image = self.pack_image(img)
        r = requests.post(
            self.url + "sdapi/v1/interrogate",
            json={"image": image, "model": "clip"},
        )
        return r.json()

    def upscale(self, img, upscaler_1="4x_foolhardy_Remacri", **kwargs):
        image = self.pack_image(img)
        payload = {"image": image, "upscaler_1": upscaler_1}
        payload.update(kwargs)
        r = requests.post(self.url + "sdapi/v1/extra-single-image", json=payload)
        if r.status_code != 200:
            print(r)
            print(r.text)
        path = self.prepare_path() + ".png"
        Path(path).write_bytes(base64.decodebytes(r.json()["image"].encode()))
        return path