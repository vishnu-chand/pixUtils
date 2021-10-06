

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
# Using Albumentations to augment bounding boxes for object detection tasks
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
## Prepare the Google Colab environment
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
#### Download images
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
Download images that are used in the notebook and save to the `images` folder in the Colab environment.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# !wget -q https://github.com/albumentations-team/albumentations_examples/archive/master.zip -O /tmp/albumentations_examples.zip
# !unzip -o -qq /tmp/albumentations_examples.zip -d /tmp/albumentations_examples
# !cp -r /tmp/albumentations_examples/albumentations_examples-master/notebooks/images .
# !echo "Images are successfully downloaded"

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
#### Install the latest version of Albumentations
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
Google Colab has an outdated version of Albumentations so we will install the latest stable version from PyPi.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# !pip install -q -U albumentations
# !echo "$(pip freeze | grep albumentations) is successfully installed"

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
## Run the example
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Import the required libraries
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# %matplotlib inline

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define functions to visualize bounding boxes and class labels on an image
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
The visualization function is based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Get an image and annotations for it
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
For this example we will use an image from the [COCO dataset](https://cocodataset.org/) that have two associated bounding boxes. The image is available at http://cocodataset.org/#explore?id=386298
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
#### Load the image from the disk
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
image = cv2.imread('images/000000386298.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
#### Define two bounding boxes with coordinates and class labels
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
Coordinates for those bounding boxes are declared using the `coco` format. Each bounding box is described using four values `[x_min, y_min, width, height]`. For the detailed description of different formats for bounding boxes coordinates, please refer to the documentation article about bounding boxes - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'cat', 18: 'dog'}

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
#### Visuaize the original image with bounding boxes
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
visualize(image, bboxes, category_ids, category_id_to_name)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define an augmentation pipeline
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
To make an augmentation pipeline that works with bounding boxes, you need to pass an instance of `BboxParams` to `Compose`. In `BboxParams` you need to specify the format of coordinates for bounding boxes and optionally a few other parameters. For the detailed description of `BboxParams` please refer to the documentation article about bounding boxes - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
We fix the random seed for visualization purposes, so the augmentation will always produce the same result. In a real computer vision pipeline, you shouldn't fix the random seed before applying a transform to the image because, in that case, the pipeline will always output the same image. The purpose of image augmentation is to use different transformations each time.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Another example
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose(
    [A.ShiftScaleRotate(p=0.5)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define a complex augmentation piepline
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### `min_area` and `min_visibility` parameters
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
The size of bounding boxes could change if you apply spatial augmentations, for example, when you crop a part of an image or when you resize an image.

`min_area` and `min_visibility` parameters control what Albumentations should do to the augmented bounding boxes if their size has changed after augmentation. The size of bounding boxes could change if you apply spatial augmentations, for example, when you crop a part of an image or when you resize an image.

`min_area` is a value in pixels. If the area of a bounding box after augmentation becomes smaller than `min_area`, Albumentations will drop that box. So the returned list of augmented bounding boxes won't contain that bounding box.

`min_visibility` is a value between 0 and 1. If the ratio of the bounding box area after augmentation to `the area of the bounding box before augmentation` becomes smaller than `min_visibility`, Albumentations will drop that box. So if the augmentation process cuts the most of the bounding box, that box won't be present in the returned list of the augmented bounding boxes.

'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define an augmentation pipeline with the default values for `min_area` and `min_visibilty`
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
If you don't pass the `min_area` and `min_visibility` parameters, Albumentations will use 0 as a default value for them.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
As you see the output contains two bounding boxes.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define an augmentation pipeline with `min_area`
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
Next, we will set the `min_area` value to 4500 pixels.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', min_area=4500, label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
The output contains only one bounding box because the area of the second bounding box became lower than 4500 pixels.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Define an augmentation pipeline with `min_visibility`
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
Finally, we will set `min_visibility` to 0.3. So if the area of the output bounding box is less than 30% of the original area, Albumentations won't return that bounding box.
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_ids']),
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
The output doesn't contain any bounding box.

Note that you can declare both the `min_area` and `min_visibility` parameters simultaneously in one `BboxParams` instance.
'''