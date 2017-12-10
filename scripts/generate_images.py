import math
import sys
sys.path.append("./")

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from utils.kjv_text import KJVTextDataset

kjv = KJVTextDataset()

# Derived from code at
# https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
def makeImage(txt, font, filename, sz):
    img = Image.new('RGB', sz, "white")
    draw = ImageDraw.Draw(img)
    draw.text((0,0), txt, (0, 0, 0), font=font)
    img.save(filename)

font_size_in = 0.25
font_size_pt = int(font_size_in * 72.0)
font_path = "/Library/Fonts/Andale Mono.ttf"    # Specific to Mac OS -- change if needed
font = ImageFont.truetype(font_path, font_size_pt)
char_height, char_width = font.getsize("A")[0:2]

chars_per_line = 32
lines_per_img = 32
image_dims_px = (char_height * chars_per_line, (font_size_pt + 3) * lines_per_img)

print("Image dimensions: (%d px x %d px)" % (image_dims_px[0], image_dims_px[1]))

text_str_per_image = kjv.image_text((chars_per_line, lines_per_img))

print("Creating %d images..." % num_imgs)
for i in range(num_imgs):
    # Print update in place
    sys.stdout.write("\r%d images processed (%d%% complete)" % (i, int(i / float(num_imgs / 100.0))))
    sys.stdout.flush()

    txt = text_str_per_image[i].rstrip('\n')    # Strip off last newline for each image; Pillow doesn't like that...
    img_filename = "images/%d.png" % i
    makeImage(txt, font, img_filename, image_dims_px)
# Insert newline to reset in-place update timer
sys.stdout.write("\r\nImage creation complete!\n")
sys.stdout.flush()
