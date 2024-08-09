import os
import PIL
import os
import os.path
from PIL import Image

folder = "/home/miki/Desktop/Miki/DeepLearning/ethiopian-currency-denomination/data/Currency/100/"
count = 1

for file in os.listdir(folder):
    source = folder + file
    name = folder + "hundred_" + str(count) + ".JPG"
    os.rename(source, name)
    count += 1

path = r'/home/miki/Desktop/Miki/DeepLearning/ethiopian-currency-denomination/data/notes/5/'
for file in os.listdir(path): 
    f_img = path+"/"+file
    img = Image.open(f_img)
    img = img.resize((1000, 1000)) #(width, height)
    img.save(f_img)