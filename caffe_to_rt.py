import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint
from PIL import Image
from tensorrt.parsers import caffeparser
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
'''INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['features']
INPUT_H = 160
INPUT_W =  160
OUTPUT_SIZE = 512
MODEL_PROTOTXT = 'facenet_deploy.prototxt'
CAFFE_MODEL = 'face_model/facenet_inc_resnet_v1_softmax_512_iter_32000.caffemodel'
DATA = '/media/ovuser/C/Data/_sideways_faces'
engine = trt.utils.caffe_to_trt_engine(G_LOGGER,MODEL_PROTOTXT,CAFFE_MODEL,1, 1 << 20,OUTPUT_LAYERS,trt.infer.DataType.FLOAT)
trt.utils.write_engine_to_file("tf_inc_resnet_v1_softmax_512.engine", engine.serialize())
'''
OUTPUT_SIZE = 512
engine = trt.utils.load_engine(G_LOGGER, "tf_inc_resnet_v1_softmax_512.engine")
import time

path = '/home/ovuser/Pictures/kgunda_60.png'
im = Image.open(path)
#imshow(np.asarray(im))
arr = np.array(im)
img = arr.ravel()


runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

assert(engine.get_nb_bindings() == 2)
#convert input data to Float32
img = img.astype(np.float32)
#create output array to receive data
output = np.empty(OUTPUT_SIZE, dtype = np.float32)

t = time.time()

d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.enqueue(1, bindings, stream.handle, None)
#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()
print ('Time taken for one image: ', time.time() - t, np.argmax(output))
