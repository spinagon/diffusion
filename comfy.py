import json
import requests
from pathlib import Path
import random

prompt_text = "cat photo, nature, lakeside"


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode("utf-8")
    r = requests.post("http://127.0.0.1:8188/prompt", json=p)
    print(r.text)
    r = requests.get("http://127.0.0.1:8188/queue")
    print(r.text)


prompt = json.loads(Path("c:/prog/1/comfy_2.json").read_text())
# set the text prompt for our positive CLIPTextEncode
prompt["1"]["inputs"]["positive"] = prompt_text

# set the seed for our KSampler node
prompt["2"]["inputs"]["noise_seed"] = random.randint(0, int(1e10))

queue_prompt(prompt)
