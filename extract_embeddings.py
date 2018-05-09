
#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import time

import caffe


def get_embeddings_caffe(input_file, model_def = '/home/ovuser/Desktop/CaffeFaceResearch/face_deploy.prototxt', pretrained_model = '/home/ovuser/Desktop/CaffeFaceResearch/face_model/facenet_inc_resnet_v1_softmax_512_iter_32000.caffemodel', gpu = True, center_only = False, images_dim = '160,160', mean_file=None, input_scale = None, raw_scale = 255.0, channel_swap = '2,1,0', ext = '.jpg'):
    #pycaffe_dir = os.path.dirname(__file__)

    image_dims = [int(s) for s in images_dim.split(',')]

    mean, channel_swap = None, None
    if mean_file:
        mean = np.load(mean_file)
    if channel_swap:
        channel_swap = [int(s) for s in channel_swap.split(',')]

    if gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory (*.jpg), or image file.
    input_file = os.path.expanduser(input_file)
    #print (input_file)
    if input_file.endswith('npy'):
        print("Loading file: %s" % input_file)
        inputs = np.load(input_file)
    elif os.path.isdir(input_file):
        print("Loading folder: %s" % input_file)
        inputs =[caffe.io.load_image(input_file+'/'+im_f) for im_f in os.listdir(input_file)]
        print (len(inputs), inputs[0].shape)
    else:
        print("Loading file: %s" % input_file)
        inputs = [caffe.io.load_image(input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not center_only)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    #print("Saving results into %s" % output_file)
    #np.save(output_file, predictions)
    return predictions

get_embeddings_caffe('/home/ovuser/Pictures/_sideways_faces', '/home/ovuser/Desktop/CaffeFaceResearch/face_deploy.prototxt', 'face_model/facenet_inc_resnet_v1_softmax_512_iter_32000.caffemodel')
