from PIL import Image
import os

path = "./INRIA_Dataset_Samples/Test/neg/"
dirs = os.listdir(path)

for item in dirs:
    #if os.path.isfile(path + dirs[0]):
    im = Image.open(path + item)
    f, e = os.path.splitext(item)
    imResize = im.resize((96, 160), Image.ANTIALIAS)
    imResize.save('./INRIA_Dataset_Samples/Test/neg_resized/' + 'resized_' + f + '.jpg', 'JPEG', quality=90)
