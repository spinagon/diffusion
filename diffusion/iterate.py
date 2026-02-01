import argparse
import sys

from matplotlib import pyplot as plt
from spinagon.diffusion import img2img


def process(img, denoise=0.1, steps=40):
    return img2img(
        img,
        denoise=denoise,
        width=512,
        height=512,
        sampler="DDIM",
        steps=steps,
        prompt="highly detailed",
        neg="in style of bad-artist",
        scale=20,
        resize_mode=1,
    )


def main(args):
    img = process(plt.imread(args.image), denoise=0.1, steps=1)
    frame0 = plt.imread(img)
    n_images = args.n
    for i in range(n_images):
        print("{:04d}/{:04d}".format(i + 1, n_images))
        new_img = plt.imread(img)
        new_img = new_img * 0.5 + frame0 * 0.5
        img = process(new_img, denoise=args.denoise, steps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=40, type=int)
    parser.add_argument("--denoise", default=0.1, type=float)
    parser.add_argument("image")
    parser.add_argument("--prompt", default="highly detailed")
    args = parser.parse_args()

    main(args)
