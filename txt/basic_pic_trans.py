"""
    先来说一下jpg图片和png图片的区别
    jpg格式:是有损图片压缩类型,可用最少的磁盘空间得到较好的图像质量
    png格式:不是压缩性,能保存透明等图

"""
from PIL import Image
import cv2 as cv
import os


def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


def JPG_PNG(JngPath):
    infile = JngPath
    outfile = os.path.splitext(infile)[0] + ".png"
    img = Image.open(infile)
    try:
        if len(img.split()) == 3:
            # prevent IOError: cannot write mode RGBA as BMP
            img = img.convert('RGBA')
            L, H = img.size
            color_0 = (255, 255, 255, 255)  # 要替换的颜色
            for h in range(H):
                for l in range(L):
                    dot = (l, h)
                    color_1 = img.getpixel(dot)
                    if color_1 == color_0:
                        color_1 = color_1[:-1] + (0,)
                        img.putpixel(dot, color_1)
            img.save(outfile)
            os.remove(JngPath)
        else:
            pass
    except Exception as e:
        print("JPG转换PNG 错误", e)

def rgb_cmyk(JngPath):
    # C = 255 - R
    # M = 255 - G
    # Y = 255 - B
    # K = 0
    infile = JngPath
    outfile = os.path.splitext(infile)[0] + ".png"
    img = Image.open(infile)
    if len(img.split()) == 3:
        # prevent IOError: cannot write mode RGBA as BMP
        img = img.convert('CMYK')
        print(img.mode)
        img.getpixel((0, 0))

if __name__ == '__main__':
    # JPG_PNG(r"E:\project\mark_tool\dataformular\1800.png")
    # PNG_JPG(r"C:\Users\lenovo\Desktop\newI\s.png")
    baspath = "E:\project\mark_tool\dataformular\原始数据2.0 - 副本"
    files = os.listdir(baspath)
    for file in files:
        if file.endswith("jpg"):
            print(file)
            JPG_PNG(os.path.join(baspath, file))
            exit()
        # JPG_PNG(r"E:\project\mark_tool\dataformular\1801.jpg")
