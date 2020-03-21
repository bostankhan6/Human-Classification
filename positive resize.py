from PIL import Image
import os

path = "./INRIA_Dataset_Samples/Test/pos/"
dirs = os.listdir(path)

for item in dirs:
    #if os.path.isfile(path + dirs[0]):
    rgba_im = Image.open(path + item)
    f, e = os.path.splitext(item)
    rgb_im = rgba_im.convert('RGB')
    imResize = rgb_im.resize((96, 160), Image.ANTIALIAS)
    imResize.save('./INRIA_Dataset_Samples/Test/pos_resized/' + 'resized_' + f + '.jpg', 'JPEG', quality=90)
