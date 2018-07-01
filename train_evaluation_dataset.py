from datasets import NovaEmotions, split_dataset
from dataGenerator import ListIterator
from model import Mini_Xception
from config import DefaultConfig

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras import optimizers

from sklearn.utils import shuffle
import os
import datetime

# parameters
opt = DefaultConfig()

# Load dataset
dataset = NovaEmotions(opt.target_emotion_map, opt.data_path)
image_filenames, labels, emotion_map = dataset.load_data()
# random sort the images and labels
image_filenames, labels = shuffle(image_filenames, labels,
                                  random_state=0)
# split the training and validation
train_dataset, validation_dataset = split_dataset(
    image_filenames, 
    labels)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = ListIterator(
    train_dataset[0],
    train_dataset[1],
    train_datagen,
    target_size=opt.target_image_size,
    color_mode='rgb',
    batch_size=opt.batch_size,
    data_format='channels_last')
test_generator = ListIterator(
    validation_dataset[0],
    validation_dataset[1],
    test_datagen,
    target_size=opt.target_image_size,
    color_mode='rgb',
    batch_size=opt.batch_size,
    data_format='channels_last')

# define model 
log_file = os.path.join(log_file_path, opt.model_name + '_' +
                        opt.dataset_name + '_' 
                        + str(datetime.datetime.now())[:19]
                        +'.log')
csv_logger = CSVLogger(log_file, append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=20, min_lr=0.00001)
save_model_name = os.path.join(opt.model_path, opt.dataset_name + '_'
                               + opt.model_name +
                               '.{epoch:02d}-{val_acc:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(save_model_name, 'val_acc', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, reduce_lr]
adam = optimizers.Adam(lr=opt.learning_rate)

model = Mini_Xception(opt.input_shape, opt.num_classes,
                      opt.learning_rate, opt.l2_regularization)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(generator=train_generator, 
                    steps_per_epoch=len(train_dataset[0]) // opt.batch_size,
                    epochs=opt.epochs,
                    callbacks=callbacks,
                    validation_data=test_generator,
                    validation_steps=len(validation_dataset[0]) // opt.batch_size)