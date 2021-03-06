IMAGE_SIZE = (224,224) # The dimensions to which all images found will be resized.
BATCH_SIZE = 16
NUMBER_EPOCHS = 500

TENSORBOARD_DIRECTORY = "../logs/simple_model/tensorboard"
TRAIN_DIRECTORY = "../data/train/"
VALID_DIRECTORY = "../data/valid/"

NUMBER_TRAIN_SAMPLES = 17500
NUMBER_VALIDATION_SAMPLES = 5000

WEIGHTS_DIRECTORY = "../weights/"

###########
# base model
###########
from keras.applications.inception_v3 import InceptionV3
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

###########
# FCN layer
###########
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
# and a logistic layer 
predictions = Dense(2, activation='softmax')(x)

###########
# complete model
###########
from keras.models import Model
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


############
# load weights
############
import os.path
model_save_path = WEIGHTS_DIRECTORY + 'inceptionv3_pretrained_weights.h5'
if os.path.exists(model_save_path) == True:
    print("Loading weights from: {}".format(model_save_path))
    model.load_weights(model_save_path)


#############
# Set the non trainable layers
#############
for layer in base_model.layers[:-13]:
    layer.trainable = False
print(len(base_model.layers))


#############
#Keras callbacks
#############
from keras.callbacks import ModelCheckpoint
# Checkpoint the model weights
checkpoint_file_path =  WEIGHTS_DIRECTORY + 'inceptionv3_pretrained_weights.h5'
checkpointer = ModelCheckpoint(filepath=checkpoint_file_path, verbose=1, save_best_only=True)
print('Setting {} as folder for checkpoints.'.format(checkpoint_file_path))
callbacks = [checkpointer, early_stop]

#############
# model optimizer
#############
OPTIMIZER_LEARNING_RATE = 1e-4
OPTIMIZER_DECAY = 1e-4
OPTIMIZER_MOMENTUM = 0.89
OPTIMIZER_NESTEROV_ENABLED = False
from keras.optimizers import SGD
optimizer = SGD(lr=OPTIMIZER_LEARNING_RATE, 
          decay=OPTIMIZER_DECAY, 
          momentum=OPTIMIZER_MOMENTUM, 
          nesterov=OPTIMIZER_NESTEROV_ENABLED)

##############
# compile
##############
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=["accuracy"])

##############
# train data generator
##############
from keras.preprocessing.image import ImageDataGenerator

## train generator with shuffle but no data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   rotation_range=20,
                                   horizontal_flip=True,
                                   vertical_flip=True)
train_batch_generator =  train_datagen.flow_from_directory(TRAIN_DIRECTORY, 
                                                 target_size = IMAGE_SIZE,
                                                 class_mode = 'categorical', 
                                                 batch_size = BATCH_SIZE)

##############
# validation data generator
##############
from keras.preprocessing.image import ImageDataGenerator

## train generator with shuffle but no data augmentation
validation_datagen = ImageDataGenerator(rescale = 1./255)

valid_batch_generator =  validation_datagen.flow_from_directory(VALID_DIRECTORY, 
                                                 target_size = IMAGE_SIZE,
                                                 class_mode = 'categorical', 
                                                 batch_size = BATCH_SIZE)

##############
# Training
##############
# fine-tune the model
hist = model.fit_generator(
        train_batch_generator,
        steps_per_epoch=NUMBER_TRAIN_SAMPLES/BATCH_SIZE,
        epochs=NUMBER_EPOCHS,  # epochs: Integer, total number of iterations on the data.
        validation_data=valid_batch_generator,
        validation_steps=NUMBER_VALIDATION_SAMPLES/BATCH_SIZE,
        callbacks=callbacks,
        verbose=1)
