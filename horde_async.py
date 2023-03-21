import asyncio
import base64
import datetime
import re
import time
from io import BytesIO
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from .aiohttp_backend import Http_backend


class Connection:
    def __init__(self, endpoint="https://stablehorde.net/api/v2", apikey=None):
        self.endpoint = endpoint
        self.apikey = apikey
        self.jobs = []

    async def init(self):
        global requests
        requests = await Http_backend.init()

    async def find_user(self):
        headers = {"apikey": self.apikey}
        r = await requests.get(self.endpoint + "/find_user", headers=headers)
        return r.json()

    async def txt2img(self, prompt, options=None, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        await job.clean()
        return result

    async def img2img(self, prompt, img, options=None, denoise=0.55, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        await job.set_image(img)
        h, w = await dimension(img)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        await job.clean()
        return result

    async def inpaint(self, prompt, img, mask=None, options=None, denoise=1, **kwargs):
        job = Job(prompt, self.apikey, self.endpoint)
        await job.set_image(img)
        job.set_mask(mask)
        job.payload["source_processing"] = "inpainting"
        h, w = await dimension(img)
        job.params["height"] = h
        job.params["width"] = w
        job.params["denoising_strength"] = denoise
        job.params.update(options or {})
        job.params.update(kwargs)
        # self.jobs.append(job)
        result = await job.run()
        await job.clean()
        return result

    async def interrogate(self, img, caption_type="caption"):
        """
        caption_type="caption" | "interrogation" | "nsfw"
        """
        job = Interrogation_job(img, self.apikey, self.endpoint, caption_type)
        result = await job.run()
        return result


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


async def dimension(img):
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
    longer = int(round(longer / shorter * 512 / 64) * 64)
    shorter = 512
    if w < h:
        width = shorter
        height = longer
    else:
        width = longer
        height = shorter
    return height, width


class Job:
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
            "nsfw": True,
            "replacement_filter": True,
        }
        self.state = "created"
        self.kind = "txt2img"
        self.source_image = None
        self.source_mask = None
        self.result = None

    async def set_image(self, image):
        self.source_image = pack_image(image)
        self.payload["source_image"] = self.source_image
        self.params["sampler_name"] = "k_euler_a"
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
            if self.result == "timeout":
                continue
            else:
                break
        self.finished_at = datetime.datetime.now()
        return self.result

    async def validate_params(self):
        if "seed" in self.params:
            self.params["seed"] = str(self.params["seed"])
        if "height" in self.params:
            self.params["height"] = round(self.params["height"] / 64) * 64
        if "width" in self.params:
            self.params["width"] = round(self.params["width"] / 64) * 64
        if "ct" in self.params:
            self.params["control_type"] = self.params.pop("ct")

    async def clean(self):
        self.source_image = None
        self.source_mask = None
        self.payload["source_image"] = None
        self.payload["source_mask"] = None

    async def generate(self):
        headers = {"apikey": self.apikey}
        for i in range(3):
            r = await requests.post(
                self.endpoint + "/generate/async", json=self.payload, headers=headers
            )
            if r.status_code != 403:
                break
            else:
                print("generate failed, ", r.text)
        try:
            uuid = r.json()["id"]
        except Exception as e:
            self.state = "failed"
            self.result = (r, r.text)
            print("generate failed, ", self.result)
            raise e
        self.uuid = uuid
        self.state = "running"

    async def status(self):
        if self.kind == "interrogate":
            r = await requests.get(self.endpoint + "/interrogate/status/" + self.uuid)
        else:
            r = await requests.get(self.endpoint + "/generate/status/" + self.uuid)
        try:
            status = r.json()
            status["prompt"] = self.prompt
            self.last_status = status
            return status
        except Exception as e:
            print("status failed")
            print(repr(e))
            print(r)
            print(r.text)

    async def await_result(self):
        wait_list = [7, 1, 1, 2, 2, 7, 10, 10, 10, 10, 6]
        waited = 0
        for i in range(100):
            index = min(i, len(wait_list) - 1)
            await asyncio.sleep(wait_list[index])
            waited += wait_list[index]
            await self.status()
            d = self.last_status
            d["waited"] = waited
            if i % 10 == 9:
                print(d)
            if "message" in d:
                print("Message in status:", d["message"])
                await asyncio.sleep(1)
                waited += 1
            if d.get("done", False) or d.get("state", None) == "done":
                self.result = d
                self.result["waited"] = waited
                self.state = "done"
                return
        print("await_result timeout", d)
        self.result = "timeout"

    def check_state(self):
        if self.result is not None:
            self.state = "done"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        self.check_state()
        return "Job {}, state: {}".format(id(self), self.state)


class Interrogation_job(Job):
    def __init__(self, img, apikey, endpoint, caption_type="caption"):
        self.source_image = img
        self.apikey = apikey
        self.endpoint = endpoint
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
        headers = {"apikey": self.apikey}
        for i in range(3):
            r = await requests.post(
                self.endpoint + "/interrogate/async", json=self.payload, headers=headers
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
