# import requests
import base64
import datetime
import pprint
import random
import re
import time
from io import BytesIO
from pathlib import Path
from thefuzz import process
import json

import imageio
import requests
from PIL import Image


class Connection:
    def __init__(
        self,
        endpoint="https://stablehorde.net/api/v2",
        apikey=None,
        agent="diff_lib:0.1:flak.yar@gmail.com",
    ):
        self.endpoint = endpoint
        self.apikey = apikey
        self.jobs = []
        self.agent = agent
        self.headers = {"apikey": self.apikey, "Client-Agent": self.agent}
        self.model_names = []

    def find_user(self):
        r = requests.get(self.endpoint + "/find_user", headers=self.headers)
        return r.json()

    def txt2img(self, prompt, denoise=0.4, options=None, **kwargs):
        job = Job(prompt, self)
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        self.jobs.append(job)
        result = job.run()
        job.clean()
        return result

    def img2img(
        self, prompt, img, options=None, denoise=0.55, autoresize=True, **kwargs
    ):
        job = Job(prompt, self)
        job.set_image(img)
        if "width" not in kwargs and "height" not in kwargs:
            if autoresize:
                h, w = dimension(img, best_size=job.best_size)
                job.params["height"] = h
                job.params["width"] = w
            else:
                h, w = dimension(img, resize=False, best_size=job.best_size)
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
        job = Job(prompt, self)
        job.set_image(img)
        if mask:
            job.set_mask(mask)
        job.payload["source_processing"] = "inpainting"
        # job.payload["models"] = ["anything_v4_inpainting"]
        job.payload["models"] = ["Deliberate Inpainting"]
        h, w = dimension(img, best_size=job.best_size)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        self.jobs.append(job)
        result = job.run()
        job.clean()
        return result

    def interrogate(self, img, caption_type="caption"):
        """
        caption, interrogation, nsfw, GFPGAN, RealESRGAN_x4plus,
        RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B,
        NMKD_Siax, 4x_AnimeSharp, CodeFormers, strip_background
        """
        job = Interrogation_job(img, self, caption_type)
        self.jobs.append(job)
        result = job.run()
        return result

    def models(self):
        result = requests.get("https://stablehorde.net/api/v2/status/models").json()
        result = sorted(result, key=lambda x: -x["count"])
        self.model_names = [x["name"] for x in result]
        return result

    def model_stats(self):
        result = requests.get("https://stablehorde.net/api/v2/stats/img/models").json()
        result = sorted(result["month"].items(), key=lambda x: -x[1])
        return result

    def heartbeat(self):
        result = requests.get("https://stablehorde.net/api/v2/status/heartbeat").json()
        return result

    def match_model(self, name, n=1):
        if len(self.model_names) == 0:
            self.models()
        matches = find_closest(name, self.model_names, n)
        if n == 1:
            return matches[0]
        else:
            return matches


def prepare_path(prompt=""):
    t = random.uniform(0.02, 0.3)
    time.sleep(t)
    path = (
        "./sd/"
        + datetime.datetime.now().isoformat().replace(":", ".")
        + "_"
        + get_slug(prompt)
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save(result, path):
    paths = []
    for i, gen in enumerate(result["generations"]):
        info = b"\n" + json.dumps(result, indent=2).encode()
        try:
            img_url = gen.pop("img")
            data = requests.get(img_url).content
            seed = gen["seed"]
            j = 0
            savepath = path + "_" + seed + "-" + str(j) + ".webp"
            while Path(savepath).exists():
                j += 1
                savepath = path + "_" + seed + "-" + str(j) + ".webp"
            paths.append(savepath)
            Path(paths[-1]).write_bytes(data + info)
        except Exception as e:
            print(repr(e))
            print(result)
    return paths


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
    if isinstance(img, Image.Image):
        data = BytesIO()
        if format is None:
            format = "JPEG"
        img.save(data, format=format)
        image = data.getvalue()
    return base64.encodebytes(image).decode()


def get_slug(prompt):
    return "_".join(re.findall("[a-zA-Z#0-9]+", prompt))[:40]


def dimension(img, resize=True, best_size=512):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    if isinstance(img, Image.Image):
        w, h = img.size
    if isinstance(img, str) or isinstance(img, Path):
        img = Path(img)
        if img.suffix == ".webp":
            w, h = Webp(img).to_image().size
        else:
            w, h = Image.open(img).size
    if resize:
        shorter = min(h, w)
        longer = max(h, w)
        longer = int(round(longer / shorter * best_size / 64) * 64)
        shorter = best_size
        if w < h:
            width = shorter
            height = longer
        else:
            width = longer
            height = shorter
    else:
        height = h
        width = w
    return height, width


class Job:
    def __init__(self, prompt, conn):
        self.prompt = prompt
        self.conn = conn
        self.params = {"sampler_name": "k_dpmpp_2m", "steps": 20, "karras": True}
        self.payload = {
            "prompt": self.prompt,
            "params": self.params,
            "models": ["Deliberate"],
            "shared": True,
            "nsfw": True,
            "replacement_filter": True,
            "trusted_workers": False,
            # "workers": ["dc0704ab-5b42-4c65-8471-561be16ad696"]
        }
        self.state = "created"
        self.kind = "txt2img"
        self.source_image = None
        self.source_mask = None
        self.result = None
        self.path = prepare_path(self.prompt)
        self.best_size = 512
        self.validate_params()

    def set_image(self, image):
        self.source_image = pack_image(image)
        self.payload["source_image"] = self.source_image
        self.params["sampler_name"] = "k_dpmpp_sde"
        self.params["steps"] = 15
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
        self.path = save(self.result, self.path)
        return self.path

    def validate_params(self):
        if "seed" in self.params:
            self.params["seed"] = str(self.params["seed"])
        if "model" in self.params:
            self.payload["models"] = [self.conn.match_model(self.params.pop("model"))]
        if "models" in self.params:
            self.payload["models"] = self.params.pop("models")
        if len(["SDXL" in x for x in self.payload.get("models", [])]) > 0:
            self.params["width"] = self.params.get("width", 1024)
            self.params["height"] = self.params.get("height", 1024)
            self.params["n"] = self.params.get("n", 2)
            self.best_size = 1024
        if "height" in self.params:
            self.params["height"] = round(self.params["height"] / 64) * 64
        if "width" in self.params:
            self.params["width"] = round(self.params["width"] / 64) * 64
        if "workers" in self.params:
            self.payload["workers"] = self.params.pop("workers")

    def clean(self):
        self.source_image = None
        self.source_mask = None
        self.payload["source_image"] = None
        self.payload["source_mask"] = None

    def generate(self):
        for i in range(3):
            r = requests.post(
                self.conn.endpoint + "/generate/async",
                json=self.payload,
                headers=self.conn.headers,
            )
            if r.status_code != 403:
                break
            else:
                print(r.text)
        try:
            uuid = r.json()["id"]
        except Exception as e:
            self.state = "failed"
            self.result = (r, r.text)
            print(self.result)
            raise e
        self.uuid = uuid
        self.state = "running"

    def status(self):
        if self.kind == "interrogate":
            r = requests.get(self.conn.endpoint + "/interrogate/status/" + self.uuid)
        else:
            r = requests.get(self.conn.endpoint + "/generate/status/" + self.uuid)
        try:
            status = r.json()
            status["prompt"] = getattr(self, "prompt", "")
            self.last_status = status
            return status
        except Exception as e:
            print(e)
            print(r)
            print(r.text)

    def await_result(self):
        wait_list = (
            [6]
            + np.diff(np.logspace(np.log10(6), np.log10(60), num=10))
            .round()
            .astype(int)
            .tolist()
            + [6] * 100
        )
        waited = 0
        for t in wait_list:
            time.sleep(t)
            waited += t
            self.status()
            d = self.last_status
            d["waited"] = waited
            if "message" in d:
                print(d["message"])
            if (d.get("done", False) or d.get("state", None) == "done") and d.get(
                "processing", 0
            ) == 0:
                self.result = d
                self.result["waited"] = waited
                self.result["payload"] = self.payload
                self.result["payload"]["source_image"] = ""
                self.result["payload"]["source_mask"] = ""
                self.state = "done"
                return

    def check_state(self):
        if self.result is not None:
            self.state = "done"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        self.check_state()
        return "<Job {}, {}, state: {}>".format(
            id(self), self.payload.get("prompt", "")[:40], self.state
        )


class Interrogation_job(Job):
    def __init__(self, img, conn, caption_type="caption"):
        self.source_image = img
        self.conn = conn
        if hasattr(caption_type, "lower"):
            caption_type = [caption_type]
        self.payload = {
            "source_image": pack_image(img),
            "forms": [{"name": x} for x in caption_type],
        }
        self.state = "created"
        self.kind = "interrogate"
        self.result = None

    def run(self):
        self.started_at = datetime.datetime.now()
        self.interrogate()
        self.await_result()
        self.finished_at = datetime.datetime.now()
        return self.result

    def interrogate(self):
        for i in range(3):
            r = requests.post(
                self.conn.endpoint + "/interrogate/async",
                json=self.payload,
                headers=self.conn.headers,
            )
            if r.status_code != 403:
                break
            else:
                print(r.text)
        try:
            uuid = r.json()["id"]
        except Exception as e:
            self.state = "failed"
            self.result = (r, r.text)
            raise e
        self.uuid = uuid
        self.state = "running"


class Webp:
    def __init__(self, name):
        from PyQt5 import QtCore, QtGui

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


def find_closest(name, variants, n=1):
    return [
        x[0]
        for x in process.extract(
            name.lower(), variants, processor=lambda x: x.lower(), limit=n
        )
    ]
