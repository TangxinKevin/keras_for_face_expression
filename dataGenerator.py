from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras_preprocessing.image import Iterator
import keras.backend as K
import numpy as np

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }


def load_img(path, grayscale=False, target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".

    #   Returns
        A PIL Image instance.

    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL image instance.
        data_format: Image data format,
            either "channels_firse" or "channels_last".

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('UNknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x





class ListIterator(Iterator):
    """Iterator capable of reading images from a filename list

    # Arguments
        ImageList:  A list of all images.
        LabelList:  the label for each image.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input
            images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read
            images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        interpolation: Interpolation method used to resample the image if 
            the target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
    """

    def __init__(self, ImageList, LabelList, image_data_generator,
                 target_size, color_mode='rgb', 
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = backend.image_data_format()
        self.image_path = ImageList
        self.label = LabelList
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; excepted "rgb" or :grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.target_size + (3, )
        else:
            self.image_shape = self.target_size + (1, )
        self.interpolation = interpolation


        # First, count the number of samples
        self.samples = len(self.image_path)
        self.num_classes = len(unique(LabelList))

        print('Found %d images belonging to %d classes' % 
              (self.samples, self.num_classes))
        return super(ListIterator, self).__init__(self.samples, 
                                                  batch_size, 
                                                  shuffle, 
                                                  seed)

    def _get_batches_od_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        batch_y = np.zeros(
            (len(index_array()),) + self.num_classes,
            dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.image_path[j]
            img = load_img(fname,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.label[j]
        return batch_x, batch_y






