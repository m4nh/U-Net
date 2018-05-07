import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import U_Net

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_list_train', dest='input_list_train', default='input_list_train.txt', help='path of the input pair, image\\timage_sem')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')

parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')

# parser.add_argument('--load_size_w', dest='load_size_w', type=int, default=2048, help='scale images to this size')
# parser.add_argument('--load_size_h', dest='load_size_h', type=int, default=1024, help='scale images to this size')

parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=1024, help='then crop to this size')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=1024, help='then crop to this size')

parser.add_argument('--num_sample',dest='num_sample',type=int,default=2975,help='number of sample to process for 1 epoch')
parser.add_argument('--num_sample_test',dest='num_sample_test',type=int,default=500,help='number of sample for validation during training or to infer during test')

parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='# of classes')

parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')

#### NOT IMPLEMENTED YET ###
parser.add_argument('--flip', dest='flip', type=bool, default=False, help='if flip the images for data argumentation')
parser.add_argument('--phase', dest='phase', help='train, test', required=True, choices=['train','test'])

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test/', help='test sample are saved here')

parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of filters in first conv layer')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = U_Net(sess, args)
        if args.phase == 'train':
            model.train(args)
        else: 
            model.test(args)

if __name__ == '__main__':
    tf.app.run()