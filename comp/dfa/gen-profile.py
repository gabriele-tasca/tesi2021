import numpy as np
from PIL import Image

def profile(data, image):
    im = Image.open(image).convert('L')
    bitmap = np.array(im)
    profile = data * bitmap/256
    return profile

# number = 10
# data = np.load("./profiles/data-"+str(number)+".npy")
# im = Image.open("./profiles/bitmap-"+str(number)+".png").convert('L')
# bitmap = np.array(im)

# profile = data * bitmap/256
# # img = Image.fromarray(np.uint8(profile*256) , 'L')
# # img.show()
# np.save("./profiles/profile-"+str(number)+".npy", profile)
