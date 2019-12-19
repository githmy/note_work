from PIL import Image
import os


def channel2RGB():
    tmpname = "37.png"
    image_path = os.path.join("..", "data", tmpname)
    image = Image.open(image_path)
    r, g, b, a = image.split()
    image = Image.merge("RGB", (r, g, b))
    image.save(image_path)


if __name__ == "__main__":
    channel2RGB()
