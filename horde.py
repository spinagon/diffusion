import requests
import time
from pathlib import Path
import datetime
import base64


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

    def generate(self, prompt, source_image=None, options=None):
        if options == None:
            options = {}
        headers = {"apikey": self.api_key}
        params = {"sampler_name": "k_dpmpp_2m", "steps": 20}
        params.update(options)
        payload = {"prompt": prompt, "params": params, "models": ["Deliberate"]}
        if source_image:
            payload["source_image"] = source_image
        r = requests.post(
            self.endpoint + "/generate/async", json=payload, headers=headers
        )
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

    def txt2img(self, prompt, options=None):
        uuid = self.generate(prompt, options)
        d = self.await_result(uuid, "image")
        return save(d)

    def img2img(self, prompt, img, options=None, denoise=0.55):
        if options is None:
            options = {}
        options["denoising_strength"] = denoise
        options["sampler_name"] = "k_euler_a"
        uuid = self.generate(prompt, source_image=pack_image(img), options=options)
        d = self.await_result(uuid, "image")
        return save(d)

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

def prepare_path():
    path = "./sd/" + datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save(result):
    data = requests.get(result["generations"][0]["img"]).content
    path = prepare_path() + ".webp"
    Path(path).write_bytes(data)
    return path


def pack_image(img):
    return base64.encodebytes(Path(img).read_bytes()).decode()
