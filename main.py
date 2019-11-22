#%matplotlib inline
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
def create_captcha(text, shear=0, size=(100, 24)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"Coval.otf", 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im);
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# matplotlib.use('TkAgg')
# get_ipython().run_line_magic('matplotlib', 'osx')
# matplotlib.use('TkAgg')
# %matplotlib osx

image = create_captcha("GENE", shear=0.5)
plt.imshow(image, cmap='Greys')

from skimage.measure import label, regionprops
def segment_image(image):
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x,start_y:end_y])
    if len(subimages) == 0:
        return [image,]
    return subimages
subimages = segment_image(image)
f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
 axes[i].imshow(subimages[i], cmap="gray")

from sklearn.utils import check_random_state
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

def generate_sample(random_state=None):
 random_state = check_random_state(random_state)
 letter = random_state.choice(letters)
 shear = random_state.choice(shear_values)
 return create_captcha(letter, shear=shear, size=(20, 20)), letters.index(letter)

image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
print("The target for this image is: {0}".format(target))

dataset, targets = zip(*(generate_sample(random_state) for i in
range(3000)))
dataset = np.array(dataset, dtype='float')
targets = np.array(targets)

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))

y = y.todense()

from skimage.transform import resize

dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for
sample in dataset])

X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.
shape[2]))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, train_size=0.9)
