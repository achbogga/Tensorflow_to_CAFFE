from inception_resnet_v1_facenet_caffe import *
inc_v4 = InceptionV4('train.txt', 'val.txt', 128, 9131)
with open('facenet_train_test.prototxt', 'w') as f:
    f.write(str(inc_v4.inception_v4_proto(16)))
