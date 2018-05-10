import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import U_Net
import postcard
import scipy.misc as misc
import utils
import cv2

parser = argparse.ArgumentParser('Inference Server')
parser.add_argument('--input_list_train', dest='input_list_train',
                    default='input_list_train.txt', help='path of the input pair, image\\timage_sem')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt',
                    help='path of the input pair for validation or testing, image\\timage_sem')

parser.add_argument('--epoch', dest='epoch', type=int,
                    default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size',
                    type=int, default=1, help='# images in batch')

parser.add_argument('--load_size_w', dest='load_size_w',
                    type=int, default=2048, help='scale images to this size')
parser.add_argument('--load_size_h', dest='load_size_h',
                    type=int, default=1024, help='scale images to this size')

parser.add_argument('--crop_size_w', dest='crop_size_w',
                    type=int, default=1024, help='then crop to this size')
parser.add_argument('--crop_size_h', dest='crop_size_h',
                    type=int, default=1024, help='then crop to this size')

parser.add_argument('--num_sample', dest='num_sample', type=int,
                    default=2975, help='number of sample to process for 1 epoch')
parser.add_argument('--num_sample_test', dest='num_sample_test', type=int, default=500,
                    help='number of sample for validation during training or to infer during test')

parser.add_argument('--input_nc', dest='input_nc', type=int,
                    default=3, help='# of input image channels')
parser.add_argument('--num_classes', dest='num_classes',
                    type=int, default=19, help='# of classes')

parser.add_argument('--niter', dest='niter', type=int,
                    default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float,
                    default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float,
                    default=0.5, help='momentum term of adam')

#### NOT IMPLEMENTED YET ###
parser.add_argument('--flip', dest='flip', type=bool, default=False,
                    help='if flip the images for data argumentation')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                    default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir',
                    default='./test/', help='test sample are saved here')

parser.add_argument('--ngf', dest='ngf', type=int, default=32,
                    help='# of filters in first conv layer')

args = parser.parse_args()


def create_lut_color(dict_color):
    zeros = np.zeros(256, np.dtype('uint8'))
    lutColor = np.dstack((zeros, zeros, zeros))
    for i in range(0, 255):
        if i in dict_color.keys():
            lutColor[0][i] = [dict_color[i][2],
                              dict_color[i][1], dict_color[i][0]]
    return np.array(lutColor)


def convertSemantic(sem,  lut_color):
    cout = np.dstack([sem, sem, sem])
    sem_color = cv2.LUT(cout.astype(np.uint8), lut_color)
    return sem_color


color_lut = create_lut_color(utils.trainId2Color)


#################################
# Initialization
#################################
inference_model, image_placeholder = None, None
session = None


def semanticCallback(header, input_image):
    global inference_model, image_placeholder, session
    print(inference_model, session)
    if inference_model is not None and session is not None:

        # TODO: FORCED RESIZE FOR UNET
        input_image = utils.centerCrop(input_image, 256, 1024)

        # IMAGE TYPE AND RANGE CHANGE
        h, w, d = input_image.shape
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float)/255.
        input_image = (input_image*2)-1

        # INFERENCE
        output, pl = session.run(
            [inference_model, image_placeholder], feed_dict={image_placeholder: input_image})

        # RESIZE OUTPUT
        pred_sem_img = misc.imresize(np.squeeze(output[0], axis=-1), (h, w))

        if header.command == "segmentation_unet":
            return "ok", pred_sem_img
        elif header.command == "segmentation_unet_color":
            colored = convertSemantic(pred_sem_img, color_lut)
            return "ok", colored
        else:
            return "error.INVALID_COMMAND", None

    return "error.DEEP_ERROR", None


socket = postcard.PostcardServer.AcceptingSocket('0.0.0.0', 8000)


def main(_):
    global inference_model, image_placeholder, session

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    session = tf.Session()
    args.phase = "inference"
    model = U_Net(session, args)
    image_placeholder, inference_model = model.buildRuntimeInferenceModel(args)

    while True:
        connection, address = socket.accept()
        server = postcard.PostcardServer(
            connection, address, data_callback=semanticCallback)


if __name__ == '__main__':
    tf.app.run()
