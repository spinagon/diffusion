import asyncio
import base64
import datetime
import pprint
import re
import time
from io import BytesIO
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from simpleeval import simple_eval
from thefuzz import process

from .aiohttp_backend import Http_backend


class Connection:
    def __init__(
        self,
        endpoint="https://aihorde.net/api/v2",
        apikey=None,
        agent="unknown:0:unknown",
    ):
        self.endpoint = endpoint
        self.apikey = apikey
        self.agent = agent
        self.jobs = []
        self.model_names = []

    async def init(self):
        global requests
        requests = await Http_backend.init()

    async def find_user(self):
        headers = {"apikey": self.apikey}
        r = await requests.get(self.endpoint + "/find_user", headers=headers)
        return r.json()

    def create_job(self, prompt):
        return Job(prompt, conn=self)

    async def txt2img(self, prompt, options=None, **kwargs):
        job = self.create_job(prompt)
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        info = job.get_info()
        await job.clean()
        return result, info

    async def img2img(self, prompt, img, options=None, denoise=0.55, **kwargs):
        job = self.create_job(prompt)
        await job.set_image(img)
        h, w = await dimension(img, best_size=job.best_size)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        info = job.get_info()
        await job.clean()
        return result, info

    async def inpaint(self, prompt, img, mask=None, options=None, denoise=1, **kwargs):
        job = self.create_job(prompt)
        await job.set_image(img)
        job.set_mask(mask)
        job.payload["source_processing"] = "inpainting"
        h, w = await dimension(img, best_size=job.best_size)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        info = job.get_info()
        await job.clean()
        return result, info

    async def interrogate(self, img, caption_type="caption"):
        """
        caption_type="caption" | "interrogation" | "nsfw"
        """
        job = Interrogation_job(img, conn=self, caption_type=caption_type)
        result = await job.run()
        return result

    async def match_model(self, name, n=1):
        if len(self.model_names) == 0:
            await self.models()
        matches = find_closest(name, self.model_names, n)
        if n == 1:
            return matches[0]
        else:
            return matches

    async def models(self):
        result = (await requests.get(f"{self.endpoint}/status/models")).json()
        result = sorted(result, key=lambda x: -x["count"])
        self.model_names = [x["name"] for x in result]
        return result

    async def close(self):
        requests.close()


def pack_image(img, format=None):
    if isinstance(img, str) or isinstance(img, Path):
        image = Path(img).read_bytes()
        if format is None:
            format = Path(img).suffix.strip(".")
    elif isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            img_8 = img
        else:
            img_8 = (img * 255).astype(np.uint8)
        data = BytesIO()
        if format is None:
            format = "png"
        imageio.imwrite(data, img_8, format=format)
        image = data.getvalue()
    elif isinstance(img, bytes):
        image = img
    return base64.encodebytes(image).decode()


async def dimension(img, best_size=512):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    if isinstance(img, str) or isinstance(img, Path):
        img = Path(img)
        w, h = Image.open(img).size
    if isinstance(img, bytes):
        img = BytesIO(img)
        w, h = Image.open(img).size
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
    return height, width


class Job:
    def __init__(self, prompt, conn):
        self.prompt = prompt
        self.params = {
            "sampler_name": "k_dpmpp_2m",
            "steps": 20,
            "karras": True,
            "seed_variation": 1,
        }
        self.payload = {
            "prompt": self.prompt,
            "params": self.params,
            "models": ["Deliberate 3.0"],
            "shared": True,
            "nsfw": True,
            "replacement_filter": True,
            "trusted_workers": False,
        }
        self.state = "created"
        self.kind = "txt2img"
        self.source_image = None
        self.source_mask = None
        self.result = None
        self.conn = conn
        self.headers = {"apikey": self.conn.apikey, "Client-Agent": self.conn.agent}
        self.best_size = 512

    async def set_image(self, image):
        self.source_image = pack_image(image)
        self.payload["source_image"] = self.source_image
        self.params["sampler_name"] = "k_dpmpp_sde"
        self.params["steps"] = 15
        if self.source_mask is None:
            self.kind = "img2img"

    async def set_mask(self, mask):
        self.source_mask = pack_image(mask)
        self.payload["source_mask"] = self.source_mask
        self.kind = "inpaint"
        self.params["denoising_strength"] = 1
        self.payload["source_processing"] = "inpainting"

    async def run(self):
        self.started_at = datetime.datetime.now()
        print(f"Started: {self.prompt}")
        await self.validate_params()
        await self.generate()
        print(f"Got uuid: {self.prompt} : {self.uuid}")
        for i in range(2):
            await self.await_result()
            if "timeout" in self.result:
                continue
            else:
                break
        self.finished_at = datetime.datetime.now()
        return self.result

    async def validate_params(self):
        if "seed" in self.params:
            self.params["seed"] = str(self.params["seed"])
        if "lora" in self.params:
            self.params["lora"] = str(self.params["lora"])
        if "denoise" in self.params:
            self.params["denoising_strength"] = float(self.params["denoise"])
            self.params.pop("denoise")
        if "model" in self.params:
            self.payload["models"] = [
                await self.conn.match_model(self.params.pop("model"))
            ]
        if "models" in self.params:
            self.payload["models"] = self.params.pop("models")
        if len([x for x in self.payload.get("models", []) if "SDXL" in x]) > 0:
            self.params["width"] = self.params.get("width", 1024)
            self.params["height"] = self.params.get("height", 1024)
            self.best_size = 1024
        if "ratio" in self.params:
            self.params["width"], self.params["height"] = size_from_ratio(
                to_float(self.params["ratio"]), self.best_size**2
            )
            self.params.pop("ratio")
        if "hires_fix" in self.params:
            h = self.params.get("height", 512)
            w = self.params.get("width", 512)
            if min(h, w) <= 512:
                self.params["height"] = h * 2
                self.params["width"] = w * 2
        if "height" in self.params:
            self.params["height"] = round(self.params["height"] / 64) * 64
        if "width" in self.params:
            self.params["width"] = round(self.params["width"] / 64) * 64
        if "control_type" in self.params:
            self.params["denoising_strength"] = 1
        if "lora" in self.params:
            lora = self.params.pop("lora").split(":")
            if len(lora) > 1:
                strength = float(lora[1])
            else:
                strength = 1
            lora = lora[0]
            if "inject" in self.params:
                inject = self.params.pop("inject")
            else:
                inject = "any"
            self.params["loras"] = [
                {
                    "name": lora,
                    "model": strength,
                    "clip": strength,
                    "inject_trigger": inject,
                }
            ]

    async def clean(self):
        self.source_image = None
        self.source_mask = None
        self.payload["source_image"] = None
        self.payload["source_mask"] = None

    async def generate(self):
        for i in range(3):
            r = await requests.post(
                self.conn.endpoint + "/generate/async",
                json=self.payload,
                headers=self.headers,
            )
            if r.status_code != 403:
                break
            else:
                print("generate failed, ", r.text)
        try:
            uuid = r.json()["id"]
        except Exception as e:
            self.state = "failed"
            self.result = {"response": r.text}
            print("generate failed, ", self.result)
            self.uuid = ""
            return
        self.uuid = uuid
        self.state = "running"

    async def check(self, kind="check"):
        if self.kind == "interrogate":
            r = await requests.get(
                self.conn.endpoint + "/interrogate/status/" + self.uuid,
                headers=self.headers,
            )
        else:
            r = await requests.get(
                self.conn.endpoint + f"/generate/{kind}/" + self.uuid,
                headers=self.headers,
            )
        try:
            status = r.json()
            if "prompt" in status:
                status["prompt"] = self.prompt
            self.last_status = status
            return status
        except Exception as e:
            print("status failed")
            print(repr(e))
            print(r)
            print(r.text)

    async def status(self):
        return await self.check("status")

    async def await_result(self):
        # wait_list = [7, 1, 1, 2, 2, 7, 10, 10, 10, 10, 6]
        wait_list = [1]
        waited = 0
        for i in range(100):
            if self.state == "failed":
                return
            index = min(i, len(wait_list) - 1)
            await asyncio.sleep(wait_list[index])
            waited += wait_list[index]
            await self.check()
            d = self.last_status
            d["waited"] = waited
            if i % 10 == 9:
                print(d)
            if "message" in d:
                print("Message in status:", d["message"])
                await asyncio.sleep(1)
                waited += 1
            if (d.get("done", False) or d.get("state", None) == "done") and d.get(
                "processing", 0
            ) == 0:
                self.result = await self.status()
                self.result["waited"] = waited
                self.state = "done"
                return
        print("await_result timeout", d)
        self.result = {"timeout": True}

    def check_state(self):
        if self.result is not None:
            self.state = "done"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        self.check_state()
        return "Job {}, state: {}".format(id(self), self.state)

    def get_info(self):
        payload = {}
        payload.update(self.payload)
        payload["source_image"] = None
        payload["source_mask"] = None
        info = [self.result, payload]
        return info


class Interrogation_job(Job):
    def __init__(self, img, conn, caption_type="caption"):
        self.source_image = img
        self.conn = conn
        self.headers = {"apikey": self.conn.apikey, "Client-Agent": self.conn.agent}
        self.caption_type = caption_type
        self.payload = {
            "source_image": pack_image(img),
            "forms": [{"name": caption_type}],
        }
        self.state = "created"
        self.kind = "interrogate"

    async def run(self):
        self.started_at = datetime.datetime.now()
        await self.interrogate()
        await self.await_result()
        self.finished_at = datetime.datetime.now()
        return self.result

    async def interrogate(self):
        for i in range(3):
            r = await requests.post(
                self.conn.endpoint + "/interrogate/async",
                json=self.payload,
                headers=self.headers,
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


def find_closest(name, variants, n=1):
    return [
        x[0]
        for x in process.extract(
            name.lower(), variants, processor=lambda x: x.lower(), limit=n
        )
    ]


def size_from_ratio(ratio, pixels):
    h = np.sqrt(pixels / ratio)
    w = pixels / h
    return w, h

def to_float(x):
    if hasattr(x, "real"):
        return float(x)
    else:
        return simple_eval(x)