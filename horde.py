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
    def __init__(self, endpoint="https://stablehorde.net/api/v2", apikey=None):
        self.endpoint = endpoint
        self.apikey = apikey
        self.jobs = []

    def find_user(self):
        headers = {"apikey": self.apikey}
        r = requests.get(self.endpoint + "/find_user", headers=headers)
        return r.json()

    def txt2img(self, prompt, options=None, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        job.params.update(options or {})
        job.params.update(kwargs)
        self.jobs.append(job)
        result = job.run()
        job.clean()
        return result

    def img2img(self, prompt, img, options=None, denoise=0.55, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        job.set_image(img)
        h, w = dimension(img)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        self.jobs.append(job)
        result = job.run()
        job.clean()
        return result

    def inpaint(self, prompt, img, mask=None, options=None, denoise=1, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        job.set_image(img)
        job.set_mask(mask)
        job.payload["source_processing"] = "inpainting"
        h, w = dimension(img)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        self.jobs.append(job)
        result = job.run()
        job.clean()
        return result

    def caption(self, img):
        uuid = self.interrogate(img, "caption")
        d = self.await_result(uuid, "interrogate")
        return d

    def interrogate(self, img, kind="caption"):
        headers = {"apikey": self.apikey}
        payload = {"forms": [{"name": kind}], "source_image": pack_image(img)}
        r = requests.post(
            self.endpoint + "/interrogate/async", json=payload, headers=headers
        )
        return r.json()["id"]


def prepare_path(prompt=""):
    time.sleep(0.02)
    path = (
        "./sd/"
        + datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
        + "_"
        + get_slug(prompt)
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save(result, path):
    try:
        img_url = result["generations"][0].pop("img")
        data = requests.get(img_url).content
        info = str(result).encode()
        seed = result["generations"][0]["seed"]
        Path(path + "_" + seed + ".webp").write_bytes(data + info)
    except Exception as e:
        print(e)
        print(result)
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
        img = Path(img)
        if img.suffix == ".webp":
            w, h = Webp(img).to_image().size
        else:
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


class Job():
    def __init__(self, prompt, apikey, endpoint):
        self.prompt = prompt
        self.apikey = apikey
        self.endpoint = endpoint
        self.params = {"sampler_name": "k_dpmpp_2m", "steps": 20}
        self.payload = {
            "prompt": self.prompt,
            "params": self.params,
            "models": ["Deliberate"],
            "shared": True,
        }
        self.state = "created"
        self.kind = "txt2img"
        self.source_image = None
        self.source_mask = None
        self.result = None
        self.path = prepare_path(self.prompt)

    def set_image(self, image):
        self.source_image = pack_image(image)
        self.payload["source_image"] = self.source_image
        self.params["sampler_name"] = "k_euler_a"
        if self.source_mask is None:
            self.kind = "img2img"

    def set_mask(self, mask):
        self.source_mask = pack_image(mask)
        self.payload["source_mask"] = self.source_mask
        self.kind = "inpaint"
        self.params["denoising_strength"] = 1
        self.payload["source_processing"] = "inpainting"

    def run(self):
        self.started_at = datetime.datetime.now()
        self.validate_params()
        self.generate()
        self.await_result()
        self.finished_at = datetime.datetime.now()
        return save(self.result, self.path)

    def validate_params(self):
        if "seed" in self.params:
            self.params["seed"] = str(self.params["seed"])

    def clean(self):
        self.source_image = None
        self.source_mask = None
        self.payload["source_image"] = None
        self.payload["source_mask"] = None

    def generate(self):
        headers = {"apikey": self.apikey}
        for i in range(3):
            r = requests.post(
                self.endpoint + "/generate/async", json=self.payload, headers=headers
            )
            if r.status_code != 403:
                break
            else:
                print(r.text())
        try:
            uuid = r.json()["id"]
        except Exception as e:
            self.state = "failed"
            self.result = (r, r.text())
            raise e
        self.uuid = uuid
        self.state = "running"

    def status(self):
        if self.kind == "interrogate":
            r = requests.get(self.endpoint + "/interrogate/status/" + self.uuid)
        else:
            r = requests.get(self.endpoint + "/generate/status/" + self.uuid)
        try:
            return r.json()
        except Exception as e:
            print(e)
            print(r)
            print(r.text)

    def await_result(self):
        wait_list = [7, 1, 1, 2, 2, 7, 10, 10] + [10] * 100
        waited = 0
        for i in range(100):
            time.sleep(wait_list[i])
            waited += wait_list[i]
            d = self.status()
            if "message" in d:
                print(message)
            if d.get("done", False):
                self.result = d
                self.result["waited"] = waited
                self.state = "done"
                return

    def check_state(self):
        if self.result is not None:
            self.state = "done"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        self.check_state()
        return "Job {}, state: {}".format(id(self), self.state)


class Webp():
    def __init__(self, name):
        if isinstance(name, Path):
            self.name = name.as_posix()
        else:
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