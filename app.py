#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load the saved model
from tensorflow.keras.models import load_model
import cv2
import numpy as np


# In[ ]:


loaded_model = load_model("yazan2.h5")

# define the image path
image_path = input("Enter the path to the image you want to classify: ")
img_size = 128
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Error: could not load image from path:", image_path)
else:
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # make a prediction with the loaded model
    pred = loaded_model.predict(img)
    class_names = ["spam", "ham"]
    class_idx = int(round(pred[0][0]))
    class_name = class_names[class_idx]

    # print the predicted class
    print("Predicted class:", class_name)

