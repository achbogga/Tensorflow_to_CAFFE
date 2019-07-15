import caffe
import json
import numpy as np
import facenet
import os
import tensorflow as tf
import re
import argparse
import sys

from facenet_caffe_definition import *

def copy_params(tf_names, caffe_names, caffe_net, model_dir = '/home/caffe/achu/frozen_models/latest_ccna.pb'):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            all_tensors = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
            all_tensor_names = [tensor.name for tensor in all_tensors]
            features_tensor_index = all_tensor_names.index('InceptionResnetV1/Bottleneck/weights:0')
            features_value = all_tensors[features_tensor_index].eval()
            caffe_net.params['features'][0].data[...] = features_value.transpose((1,0))
            features_bn_1_index = all_tensor_names.index('InceptionResnetV1/Bottleneck/BatchNorm/moving_mean:0')
            features_bn_1_value = all_tensors[features_bn_1_index].eval()
            
            features_bn_2_index = all_tensor_names.index('InceptionResnetV1/Bottleneck/BatchNorm/moving_variance:0')
            features_bn_2_value = all_tensors[features_bn_2_index].eval() + 0.001
            caffe_net.params['features_bn'][1].data[...] = features_bn_2_value
            caffe_net.params['features_bn'][2].data[...] = np.ones(caffe_net.params['features_bn'][2].data.shape)
            features_bn_beta_index = all_tensor_names.index('InceptionResnetV1/Bottleneck/BatchNorm/beta:0')
            features_bn_beta_value = all_tensors[features_bn_beta_index].eval()
            
            new_mu_value = np.subtract(features_bn_1_value, np.multiply(features_bn_beta_value, np.sqrt(features_bn_2_value)))
            
            caffe_net.params['features_bn'][0].data[...] = new_mu_value
            
            for i in range(len(caffe_names)):
                caffe_name = caffe_names[i]
                tf_conv_layer_name = tf_names[i]+'/weights:0'
                tf_conv_name_index = all_tensor_names.index(tf_conv_layer_name)
                tf_conv_layer_tensor = all_tensors[tf_conv_name_index]
                print ('Copying ' + tf_conv_layer_tensor.name+' into '+caffe_name)
                conv_layer_value = tf_conv_layer_tensor.eval()
                print ('TF conv_layer shape: ', conv_layer_value.shape)
                print ('Caffe conv_layer shape: ', caffe_net.params[caffe_name][0].data.shape)
                #if (caffe_net.params[caffe_name][0].data.shape == conv_layer_value.transpose((3,2,1,0)).shape):
                #    caffe_net.params[caffe_name][0].data[...] = conv_layer_value.transpose((3,2,1,0))
                if (caffe_net.params[caffe_name][0].data.shape == conv_layer_value.transpose((3,2,0,1)).shape):
                    caffe_net.params[caffe_name][0].data[...] = conv_layer_value.transpose((3,2,0,1))
                else:
                    print ('Shape mismatch: ', caffe_name, caffe_net.params[caffe_name][0].data.shape, conv_layer_value.shape)
                
                if (re.match('InceptionResnetV1/Repeat_1/block17_\d+/Conv2d_1x1', tf_names[i].encode("ascii")) is None) and (re.match('InceptionResnetV1/Repeat/block35_\d/Conv2d_1x1', tf_names[i].encode("ascii")) is None) and (re.match('InceptionResnetV1/Block8/Conv2d_1x1', tf_names[i].encode("ascii")) is None) and (re.match('InceptionResnetV1/Repeat_2/block8_\d/Conv2d_1x1', tf_names[i].encode("ascii")) is None):
                    print (tf_names[i], ' copying BN layer params')
                    tf_bn_layer_name_1 = tf_names[i]+'/BatchNorm/moving_mean:0'
                    tf_bn_layer_name_2 = tf_names[i]+'/BatchNorm/moving_variance:0' #+0.001
                    #tf_bn_layer_name_3 to be set to one
                    #tf_scale_layer_name_1 to be set to one Gamma
                    tf_scale_layer_name_2 = tf_names[i]+'/BatchNorm/beta:0'
                    caffe_bn_layer_name = caffe_name + '_bn'
                    caffe_scale_layer_name = caffe_name + '_scale'


                    tf_bn_name_index_1 = all_tensor_names.index(tf_bn_layer_name_1)
                    tf_bn_name_index_2 = all_tensor_names.index(tf_bn_layer_name_2)
                    tf_scale_name_index_2 = all_tensor_names.index(tf_scale_layer_name_2)


                    tf_bn_layer_1_tensor = all_tensors[tf_bn_name_index_1]
                    tf_bn_layer_2_tensor = all_tensors[tf_bn_name_index_2]
                    tf_scale_layer_2_tensor = all_tensors[tf_scale_name_index_2]


                    bn_layer_1_value = tf_bn_layer_1_tensor.eval() #Mu
                    bn_layer_2_value = tf_bn_layer_2_tensor.eval() + 0.001 #Sigma
                    #print (bn_layer_2_value)
                    bn_layer_3_value = np.ones(caffe_net.params[caffe_bn_layer_name][2].data.shape)
                    #scale_layer_1_value = np.ones(caffe_net.params[caffe_scale_layer_name][0].data.shape)
                    scale_layer_2_value = tf_scale_layer_2_tensor.eval() #Beta
                    
                    new_mu_value = np.subtract(bn_layer_1_value, np.multiply(scale_layer_2_value, np.sqrt(bn_layer_2_value)))
                    #print (new_mu_value)
                    bn_layer_1_value = new_mu_value

                    #Actual copy operations

                    #print (caffe_net.params[caffe_bn_layer_name][0].data.shape, bn_layer_1_value.shape)
                    caffe_net.params[caffe_bn_layer_name][0].data[...] = bn_layer_1_value

                    #print (caffe_net.params[caffe_bn_layer_name][1].data.shape, bn_layer_2_value.shape)
                    caffe_net.params[caffe_bn_layer_name][1].data[...] = bn_layer_2_value

                    #print (caffe_net.params[caffe_bn_layer_name][2].data.shape, bn_layer_3_value.shape)
                    caffe_net.params[caffe_bn_layer_name][2].data[...] = bn_layer_3_value

                    #print (caffe_net.params[caffe_scale_layer_name][0].data.shape, scale_layer_1_value.shape)
                    #caffe_net.params[caffe_scale_layer_name][0].data[...] = scale_layer_1_value

                    #print (caffe_net.params[caffe_scale_layer_name][1].data.shape, scale_layer_2_value.shape)
                    #caffe_net.params[caffe_scale_layer_name][1].data[...] = scale_layer_2_value
                else:
                    tf_conv_layer_biases = tf_names[i]+'/biases:0'
                    tf_conv_biases_index = all_tensor_names.index(tf_conv_layer_biases)
                    tf_conv_layer_biases_tensor = all_tensors[tf_conv_biases_index]
                    print ('Copying Biases ' + tf_conv_layer_biases_tensor.name+' into '+caffe_name)
                    conv_layer_biases_value = tf_conv_layer_biases_tensor.eval()
                    caffe_net.params[caffe_name][1].data[...] = conv_layer_biases_value
            
            return caffe_net

def main(args):
	with open(args.correspondances_json_file, 'r') as fp:
	    correspondances = json.load(fp)
	if args.redefine_network!='InceptionResnetV1':
		print ('redefining network: '+args.redefine_network)
		inc_resnet_v1 = InceptionV4()
		network_definition = inc_resnet_v1.inception_v4_proto(64, 'TEST')
		with open(args.network_definition_file, 'w') as fp:
		    fp.writelines(str(network_definition))

	caffe_net = caffe.Net(args.network_definition_file, caffe.TEST)

	new_caffe_net = copy_params(list(correspondances.keys()), list(correspondances.values()), caffe_net, model_dir = args.model_dir)
	new_caffe_net.save(args.output_file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported caffe model (.caffemodel)')
    parser.add_argument('--correspondances_json_file', type=str,
        help='The layer name correspondances (tf_to_caffe) json file', default='tf_2_caffe_correspondances.json')
    parser.add_argument('--redefine_network', type = str,
    	help='Flag whether to redefine the network architecture (default:InceptionResnetV1)', default='InceptionResnetV1')
    parser.add_argument('--network_definition_file', type = str,
    	help='The network_definition_file path.', default='/home/caffe/achu/tf_2_caffe/facenet_test_clobotics.prototxt')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))