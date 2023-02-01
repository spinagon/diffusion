from pathlib import Path
import argparse
from .main import Connection, default_url


def main(args):
    conn = Connection(default_url)
    images = list(Path(args.folder).glob("*.jpg"))
    if not images:
        images = list(Path(args.folder).glob("*.png"))
    for image in images:
        caption = conn.interrogate(image.as_posix())
        (image.parent / (image.stem + ".txt")).write_text(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default=".")
    args = parser.parse_args()

    main(args)
