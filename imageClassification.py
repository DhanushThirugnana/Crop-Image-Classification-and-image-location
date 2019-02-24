
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import os
import urllib										### Request image from URL
import cv2  										### For Image Processing
import exifread  									### To extract details like GPS information from the image

from keras.models import Sequential
from keras.layers import Dense						
from keras.layers import Dropout
from keras.layers import Activation, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

from gmplot import gmplot							### Geolocation Visualization


# In[2]:


#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


### Path of the training data
training_path = "imgs/"
### Path of the validation data
validation_path = "valid/"


# In[4]:


model = Sequential()

##### Applying the convolution to the image

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(224,224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##### Flattening the resultant image to produce a fully connected feedforward network
model.add(Flatten()) 

model.add(Dense(64)) 
model.add(Activation('relu')) 
##### Dropout is used to avoid overfitting
model.add(Dropout(0.5)) 
model.add(Dense(5)) 
model.add(Activation('sigmoid')) 

##### Stochastic Gradient Descent 
stocGD = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=stocGD, metrics=['accuracy'])



# In[5]:

##### Prints the summary of the model created which includes details regarding layers of CNN model
print(model.summary())


# In[6]:


size = 224
width, heigth = size, size
epoch = 20
batch = 16
##### ImageDataGenerator is used for creating batches of images with augmentation, for increasing the accuracy while daeling with realtime images.

training_data_generator = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)


training_generator = training_data_generator.flow_from_directory(training_path,target_size=(224, 224),batch_size=batch)
validation_generator = validation_data_generator.flow_from_directory(validation_path,target_size=(224, 224),batch_size=batch)



# In[66]:

#####Training the model with the given data
model.fit_generator(training_generator,steps_per_epoch=len(training_generator.filenames)//batch,epochs=epoch)


# In[10]:

#####Extracting image from an URL
def img_from_url(url):
    urlreq = urllib.request.urlopen(url)
    img = np.asarray(bytearray(urlreq.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[11]:


##Input is Image URL 
img1 = img_from_url("picture.jpg")

###When input is downloaded image, please provide the path the input image file
img1 = cv2.imread("picture.jpg")                     ## Comment it incase or image URL
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)		 ## Comment it incase or image URL


# In[12]:

##### Display the test image 
plt.imshow(img1)
plt.show()


# In[13]:

##### Adjusting the size of the image
img1 = cv2.resize(img1,(224,224))


# In[14]:

#Check the shape of the image
img1.shape


# In[15]:


plt.imshow(img1)
plt.show()


# In[16]:

##### Normalizing the image 
img = img1.astype('float32')
img = img/255


# In[17]:


img.shape


# In[18]:

#### Appending a dimension to the image, which represents the number of samples
img = np.expand_dims(img,axis=0)
img.shape


# In[19]:

### testing on the image 
match_probs = model.predict(img)


# In[20]:

match_probs


# In[21]:

##### Returns the index with the highest value
def classify(probs):
    mx=max(probs)
    for i in range(len(probs)):
        if mx==probs[i]:
            return i
        


# In[22]:


c=classify(match_probs[0])


# In[23]:

##### Classes of crop 
array=["alfalfa","barley","corn","soyabean","wheat"]
array[c]


# In[25]:

##### Validating the model
val1 = model.fit_generator(validation_generator,steps_per_epoch=len(validation_generator.filenames) // batch,epochs=epoch)


# In[27]:

##### Plot the Loss and Accuracy against the Number of Epochs
plt.style.use("ggplot")
plt.figure()
N = epoch
#print(val1.history)
plt.plot(np.arange(0, N), val1.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), val1.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Number of epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")


# In[29]:

##### Converting the latitude/longitude values to degrees
def degree(coord):
    coor_a = float(coord.values[0].num)/float(coord.values[0].den)
    coor_b = float(coord.values[1].num)/float(coord.values[1].den)
    coor_c = float(coord.values[2].num)/float(coord.values[2].den)

    return coor_a+(coor_b/60.0)+(coor_c/3600.0)

##### Extract the GPS Latitude and Longitude Information from the images
def GeoLoc(imgloc):
    with open(imgloc, 'rb') as f:
        tags = exifread.process_file(f)
        GPSLat = tags.get('GPS GPSLatitude')
        GPSLat_ref = tags.get('GPS GPSLatitudeRef')
        GPSLong = tags.get('GPS GPSLongitude')
        GPSLong_ref = tags.get('GPS GPSLongitudeRef')
        lat_deg = degree(GPSLat)
        long_deg = degree(GPSLong)
        if GPSLat_ref.values != 'N':
            lat_deg *= -1
        if GPSLong_ref.values != 'E':            
            long_deg *= -1
        return {'latitude': lat_deg, 'longitude': long_deg}
    return {}


# In[30]:

##### Filtering the files with .jpeg/.jpg extensions and the ones that have GPS information
g=[]
for dirs, subdir, imgs in os.walk(training_path):
    for img in imgs:
        if os.path.splitext(img)[1].lower() in ('.jpg', '.jpeg'):
            f = open(dirs+'/'+img, 'rb')
            tags = exifread.process_file(f)
            if tags.get('GPS GPSLatitude') != None and tags.get('GPS GPSLongitude') != None:
                g.append(GeoLoc(dirs+'/'+img))
#print(g)


# In[31]:

##### Latitude and Longitude information stored in different array for easy access
lat=[]
long=[]
for i in g:
    lat.append(i["latitude"])
    long.append(i["longitude"])
#print(lat)
#print(long)


# In[32]:

##### Initializes the Google Map with the first image GPS Location. 
map1 = gmplot.GoogleMapPlotter(lat[0],long[0],10)


# In[33]:

##### Plots the locations of various images with GPS information on the Google Map
map1.scatter( lat, long, '#FF0000',size = 100, marker = False )
map1.draw("map1.html")

