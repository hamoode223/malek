#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models


# In[4]:


data_dir = "C:\\Users\\DELL\\Desktop\\dataset"
classes = ["spam_file", "ham_file"]
img_size = 128


# In[6]:


def create_data():
    dataset = []
    for category in classes:
        path = os.path.join(data_dir, category)
        class_num = classes.index(category) #عشان احوله ل نمباي
        for img in os.listdir(path):# بدور  عالدايركتوري وبحولها ل ليست
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                img_arr = cv2.resize(img_arr, (img_size, img_size))
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                dataset.append([img_arr, class_num]) #بخزنهم
            except Exception as e:
                pass
    return dataset


# In[7]:


dataset = create_data()

X = []
y = []

for features, label in dataset:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)#x input , y output


# In[8]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])



# In[9]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=32, batch_size=5)
model.save("yazan2.h5")

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", conf_matrix)


# In[ ]:


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:
