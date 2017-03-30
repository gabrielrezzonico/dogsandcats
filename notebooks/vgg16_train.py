IMAGE_SIZE = (180,202) # The dimensions to which all images found will be resized.
BATCH_SIZE = 16
NUMBER_EPOCHS = 8

TENSORBOARD_DIRECTORY = "../logs/simple_model/tensorboard"
TRAIN_DIRECTORY = "../data/train/"
VALID_DIRECTORY = "../data/valid/"

NUMBER_TRAIN_SAMPLES = 17500
NUMBER_VALIDATION_SAMPLES = 5000

PRECOMPUTED_DIRECTORY = "../precomputed/vgg16/"

WEIGHTS_DIRECTORY = "../weights/"

###########
# base model
###########
from keras.applications.vgg16 import VGG16
# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

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

#############
# Set the non trainable layers
#############
TRAINABLE_LAST_LAYERS = 0
assert TRAINABLE_LAST_LAYERS >= 0
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
if TRAINABLE_LAST_LAYERS == 0:
    for layer in base_model.layers:
        layer.trainable = False
    print(len(base_model.layers))
else:
    for layer in base_model.layers[:-TRAINABLE_LAST_LAYERS]:
        layer.trainable = False
    print(len(base_model.layers[:-TRAINABLE_LAST_LAYERS]))


#############
#Keras callbacks
#############
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
# Early stop in case of getting worse
early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 0)
callbacks = [early_stop]#, tensorboard_logger]

#############
# model optimizer
#############
OPTIMIZER_LEARNING_RATE = 1e-2
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
                                   zoom_range=0.2,
                                   horizontal_flip=True)
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


##############
# save weights
##############
model_save_path = WEIGHTS_DIRECTORY + 'vgg16_pretrained_v1.h5'
print('Saving TOP (FCN) weigths to ', model_save_path)
model.save(model_save_path)
