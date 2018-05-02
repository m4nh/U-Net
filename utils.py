from collections import namedtuple
import numpy as np
import tensorflow as tf


Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]

labels_encoding = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'other'                , 255 ,     255 , 'void'   , 0       , False        , False       , (0  ,0  ,0  ) ),
    Label(  'bicycle'              , 0 ,         0 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bridge'               , 1 ,         1 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 2 ,         2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'bus'                  , 3 ,         3 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 4 ,         4 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'fence'                , 5 ,         5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'motorcycle'           , 6 ,         6 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'person'               , 7 ,         7 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'pole'                 , 8 ,         8 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'rider'                , 9 ,         9 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'road'                 , 10 ,        10 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             , 11 ,       11 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'landscape'            , 12 ,       12 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'vegetation'           , 13 ,       13 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'traffic light'        , 14 ,       14 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 15 ,       15 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'truck'                , 16 ,       16 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'tunnel'               , 17 ,       17 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 18 ,       18 , 'construction'    , 2       , False        , False        , (102,102,156) )
]

trainId2Color = { label.trainId : label.color for label in labels }

def color(pred_sem, id2color=trainId2Color):
    p = tf.squeeze(tf.cast(pred_sem,tf.uint8), axis = -1)
    p = tf.stack([p,p,p],axis=-1)
    m = tf.zeros_like(p)
    for i in range(len(trainId2Color.keys()) - 1):
        mi = tf.multiply(tf.ones_like(p), trainId2Color[i])
        m = tf.where(tf.equal(p,i), mi, m)
    return m
