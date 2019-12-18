
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
#from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.callbacks import EarlyStopping,  ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam
import os
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.keras.optimizers import RMSprop

def identity_block(input_tensor, kernel_size, filters):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
  
 
    filters1, filters2, filters3 = filters
 
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
 
    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
 
 
    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
 
    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
 
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)
 
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(num_classes, input_shape):
    img_input = layers.Input(shape=input_shape)
 
    if backend.image_data_format() == 'channels_first':
        x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                          name='transpose')(img_input)
        bn_axis = 1
    else:  # channels_last
        x = img_input
        bn_axis = 3
 
    # Conv1 (7x7,64,stride=2)
    #x = layers.ZeroPadding2D(padding=(3, 3))(x)
 
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
 
    # 3x3 max pool,stride=2
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
 
    # Conv2_x
 
    # 1×1, 64
    # 3×3, 64
    # 1×1, 256
 
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    ########x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
 
    # Conv3_x
    #
    # 1×1, 128
    # 3×3, 128
    # 1×1, 512
 
    x = conv_block(x, 3, [128, 128, 512])
    ######## x = identity_block(x, 3, [128, 128, 512])
    ########x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
 
    # Conv4_x
    # 1×1, 256
    # 3×3, 256
    # 1×1, 1024
    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    ########x = identity_block(x, 3, [256, 256, 1024])
    ########x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
 
    # 1×1, 512
    # 3×3, 512
    # 1×1, 2048
    x = conv_block(x, 3, [512, 512, 2048])
    ########x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
 
    # average pool, 1000-d fc, softmax
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        2, activation='softmax')(x)
 
    # Create model.
    return models.Model(img_input, x, name='resnet50')


#resnet_weights_path = r'C:\Users\iit\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

train_data_generator = ImageDataGenerator(
       rescale=1./255
       )
test_data_generator = ImageDataGenerator(rescale=1./255)
image_size = 224
batch_size = 32


train_generator = train_data_generator .flow_from_directory(
        r'C:\data\chest_xray\train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_data_generator.flow_from_directory(
        r'C:\data\chest_xray\val',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')
num_classes = len(train_generator.class_indices)
filepath = "C:\data\chest_xray\pneumonia-{epoch:02d}-{val_acc:.4f}-{val_loss:.5f}.hdf5"
early_Stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 6)
check_Point = ModelCheckpoint(filepath, monitor='val_acc', mode='auto', period=1, verbose=1)
tbCallBack = TensorBoard(log_dir='.\Graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True)

model = resnet50(num_classes, (224, 224, 3))
model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=1e-4),
                metrics=['acc'])
model.fit_generator(  
        train_generator,
        steps_per_epoch=100,
        callbacks=[early_Stop, check_Point, tbCallBack],
        verbose=1,
        validation_data = validation_generator, 
        epochs=100)
model.save("pneumonia.h5")







from tensorflow.python.keras.preprocessing.image import load_img
from skimage.transform import rescale, resize
from tensorflow.python.keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from tensorflow.python.keras.models import load_model
model = load_model("C:\data\chest_xray\pneumonia-07-0.8622-0.53910.hdf5")

import os

path = r'C:\data\chest_xray\test\PNEUMONIA'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))
normal = 0
pneumonia = 0

# load the image
for file in files:
    img = load_img(file, target_size=(224,224))
    #Show
    #imgplot = pyplot.imshow(img)
    #pyplot.show()
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    
    #img = img.astype('float32')
    #img = rescale(img, 1./255, anti_aliasing=False)
    clas = model.predict(img/255.0)

    if clas[0][0] > clas[0][1]:
        print("Normal")
        normal+=1
    else:
        print("Pneumonia")
        pneumonia+=1
print("NORMAL: ", normal)
print("PNEUMONIA: ", pneumonia)

if normal>pneumonia:
    print("ACCURACY: ", normal/(pneumonia+normal))
else:
    print("ACCURACY: ", pneumonia/(pneumonia+normal))