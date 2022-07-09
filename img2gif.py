from PIL import Image
import cv2
import os
import imageio

result = []
paths = 'D:/save/1004_212707'
images = []
for (root, dir, files) in os.walk(paths):
    for file in files:
        if str(file).endswith(".jpg"):
            path = os.path.join(root, file)
            img = cv2.imread(path)
            img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result.append(img)
            name = path.split(".jpg")[0]
            cv2.imwrite(f'./pngs/{name}.png', img)




    #         path = os.path.join(root, file)
    #         images.append(imageio.imread(path))
    # imageio.mimsave('D:/save/1007_180305/1007_180305.gif', images)



import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg

imageio.mimsave('D:/save/1004_212707/1004_212707.gif', result)