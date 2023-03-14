import requests
import time
from pathlib import Path
import datetime
import base64
from io import BytesIO
import re
from PIL import Image
from PyQt5 import QtCore, QtGui
import imageio


class Connection:
    def __init__(self, endpoint="https://stablehorde.net/api/v2", api_key=None):
        self.endpoint = endpoint
        self.api_key = api_key

    def status(self, uuid, kind):
        if kind == "image":
            r = requests.get(self.endpoint + "/generate/status/" + str(uuid))
        elif kind == "interrogate":
            r = requests.get(self.endpoint + "/interrogate/status/" + str(uuid))
        else:
            raise Exception("Unknown kind of status: {}".format(kind))
        try:
            return r.json()
        except Exception as e:
            print(e)
            print(r)
            print(r.text)

    def generate(self, prompt, options=None, **kwargs):
        if options is None:
            options = {}
        headers = {"apikey": self.api_key}
        if "seed" in options:
            options["seed"] = str(options["seed"])
        params = {"sampler_name": "k_dpmpp_2m", "steps": 20}
        params.update(options)
        payload = {
            "prompt": prompt,
            "params": params,
            "models": ["Deliberate"],
            "shared": True,
        }
        payload.update(kwargs)
        for i in range(3):
            r = requests.post(
                self.endpoint + "/generate/async", json=payload, headers=headers
            )
            if r.status_code != 403:
                break
            else:
                print(r.text())
        try:
            uuid = r.json()["id"]
        except Exception as e:
            print(e)
            print(r)
            print(r.text)
        return uuid

    def find_user(self):
        headers = {"apikey": self.api_key}
        r = requests.get(self.endpoint + "/find_user", headers=headers)
        return r.json()

    def txt2img(self, prompt, options=None, **kwargs):
        return self.common2img(prompt, img=None, options=options, **kwargs)

    def img2img(self, prompt, img, options=None, denoise=0.55, **kwargs):
        h, w = dimension(img)
        dim = {"height": h, "width": w}
        if options is None:
            options = {}
        options.update(dim)
        return self.common2img(
            prompt, img=img, options=options, denoising_strength=denoise, **kwargs
        )

    def inpaint(self, prompt, img, mask=None, options=None, denoise=1, **kwargs):
        h, w = dimension(img)
        dim = {"height": h, "width": w}
        if options is None:
            options = {}
        options.update(dim)
        return self.common2img(
            prompt,
            img=img,
            mask=mask,
            options=options,
            denoising_strength=denoise,
            **kwargs
        )

    def common2img(self, prompt, img=None, mask=None, options=None, **kwargs):
        path = prepare_path(prompt)
        if options is None:
            options = {}
        if img is not None:
            options["sampler_name"] = "k_euler_a"
        options.update(kwargs)
        if img is not None:
            if mask is not None:
                uuid = self.generate(
                    prompt,
                    source_image=pack_image(img),
                    source_mask=pack_image(mask),
                    source_processing="inpainting",
                    options=options,
                )
            else:
                uuid = self.generate(prompt, source_image=pack_image(img), options=options)
        else:
            uuid = self.generate(prompt, options=options)
        d = self.await_result(uuid, "image")
        return save(d, path)

    def caption(self, img):
        uuid = self.interrogate(img, "caption")
        d = self.await_result(uuid, "interrogate")
        return d

    def interrogate(self, img, kind="caption"):
        headers = {"apikey": self.api_key}
        payload = {"forms": [{"name": kind}], "source_image": pack_image(img)}
        r = requests.post(
            self.endpoint + "/interrogate/async", json=payload, headers=headers
        )
        return r.json()["id"]

    def await_result(self, uuid, kind):
        wait_list = [5, 2, 1, 1, 2, 3, 4] + [15] * 100
        for i in range(100):
            time.sleep(wait_list[i])
            d = self.status(uuid, kind)
            if "message" in d:
                print(message)
            if d.get("finished", 0) == 1 or d.get("state", 0) == "done":
                return d


def prepare_path(prompt=""):
    path = (
        "./sd/"
        + datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
        + "_"
        + get_slug(prompt)
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save(result, path):
    img_url = result["generations"][0].pop("img")
    data = requests.get(img_url).content
    info = str(result).encode()
    seed = result["generations"][0]["seed"]
    Path(path + "_" + seed + ".webp").write_bytes(data + info)
    return path


def pack_image(img, format=None):
    if isinstance(img, str) or isinstance(img, Path):
        image = Path(img).read_bytes()
        if format is None:
            format = Path(img).suffix.strip(".")
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            img_8 = img
        else:
            img_8 = (img * 255).astype(np.uint8)
        data = BytesIO()
        if format is None:
            format = "png"
        imageio.imwrite(data, img_8, format=format)
        image = data.getvalue()
    return base64.encodebytes(image).decode()


def get_slug(prompt):
    return "_".join(re.findall("[a-zA-Z#0-9]+", prompt))[:40]


def webp_to_image(name):
    img = QtGui.QImage(name)
    buf = QtCore.QBuffer()
    img.save(buf, format="png")
    bb = BytesIO(buf.data())
    return Image.open(bb)


def dimension(img):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    if isinstance(img, str) or isinstance(img, Path):
        w, h = Image.open(img).size
    shorter = min(h, w)
    longer = max(h, w)
    longer = int(round(longer / shorter * 512 / 64) * 64)
    shorter = 512
    if w < h:
        width = shorter
        height = longer
    else:
        width = longer
        height = shorter
    return height, width


class Webp():
    def __init__(self, name):
        self.name = name
        self.data = None

    def read_data(self):
        if self.data is None:
            qimage = QtGui.QImage(self.name)
            buf = QtCore.QBuffer()
            qimage.save(buf, format="png")
            self.data = BytesIO(buf.data())
        self.data.seek(0)

    def to_image(self):
        self.read_data()
        return Image.open(self.data)

    def to_array(self):
        self.read_data()
        return imageio.imread(self.data)