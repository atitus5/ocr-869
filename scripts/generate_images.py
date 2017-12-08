import math
import sys
sys.path.append("./")

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from utils.kjv_text import KJVTextDataset

kjv = KJVTextDataset()
text_str = kjv.full_text

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

num_lines = int(math.ceil(len(text_str) / float(chars_per_line)))
num_imgs = int(math.floor(num_lines / float(lines_per_img)))    # Use floor to cut out the last partial image
text_str_per_line = [text_str[i * chars_per_line:(i + 1) * chars_per_line] + "\n" for i in range(num_lines)]
text_str_per_image = ["".join(text_str_per_line[i * lines_per_img:(i + 1) * lines_per_img]) for i in range(num_imgs)]

print("Creating label file (text)...")
with open("images/labels.txt", 'w') as labels:
    for txt in text_str_per_image:
        # Remove newlines in label
        txt_label = txt.replace("\n", "")
        labels.write(txt_label + "\n")
print("Done creating label file (text).")

print("Creating label file (Numpy integer labels)...")
label_mat = np.zeros((num_imgs, chars_per_line * lines_per_img), dtype=int)
for i in range(num_imgs):
    # Remove newlines in label
    txt = text_str_per_image[i]
    txt_label = txt.replace("\n", "")
    label_integers = [kjv.char_to_int[x] for x in txt_label]
    label_mat[i, :] = label_integers
np.save("images/labels.npy", label_mat)
print("Done creating label file (Numpy integer labels).")


'''
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
'''
