import sys
sys.path.append('/home/suraj/suraj/novus/novus_pilot_deep_learning/ssd_keras_updated/')
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,TerminateOnNaN, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from models.keras_mobilenet_feature_fuse import ssd_300

from utils_1.keras_ssd_loss import SSDLoss, FocalLoss, weightedSSDLoss, weightedFocalLoss
from utils_1.keras_layer_AnchorBoxes import AnchorBoxes
from utils_1.keras_layer_L2Normalization import L2Normalization
from utils_1.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from utils_1.ssd_batch_generator import BatchGenerator
#from keras.utils.training_utils import multi_gpu_model
import os
import keras
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
batch_size = 1
subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
swap_channels = True  # The color channel order in the original SSD is BGR
n_classes = 2  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

confidence_thresh=0.15
iou_threshold=0.45
top_k=200
nms_max_output_size=400

def getCombinedModel():

    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(layers = "all",
                    mode = 'inference',
                    image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    limit_boxes=limit_boxes,
                    variances=variances,
                    coords=coords,
                    normalize_coords=normalize_coords,
                    subtract_mean=subtract_mean,
                    divide_by_stddev=None)
    return model



def display_prediction(y_pred, y_pred_fcn, test_image_org, confidence_threshold = 0.5):
    #print('in display')
    #plt.figure(figsize=(20,12))
    #plt.imshow(test_image_org)
    #print(y_pred_fcn.shape)
    #y_pred_fcn = y_pred_fcn.squeeze(0)
    #print(y_pred_fcn.shape)
    #y_pred_fcn = cv2.resize(y_pred_fcn, (test_image_org.shape[1], test_image_org.shape[0]));print("***",test_image_org.shape, y_pred_fcn.shape)
    
    #print('resize faturemap', y_pred_fcn.shape)
    #print(test_image_org[..., 1].shape)
    #print(y_pred_fcn)
    #np.savetxt("keras_seg_output.csv", y_pred_fcn, delimiter=",")
    #test_image_org[..., 1][y_pred_fcn > 0.7] = 255
    
    confidence_threshold = confidence_threshold
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])
    
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background','ORU','VRU']

    #plt.figure(figsize=(20,12))
    #plt.imshow(test_image_org)

    current_axis = plt.gca()

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * test_image_org.shape[1] / img_width
        ymin = box[3] * test_image_org.shape[0] / img_height
        xmax = box[4] * test_image_org.shape[1] / img_width
        ymax = box[5] * test_image_org.shape[0] / img_height
        color = colors[int(box[0])]
        #label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        label = str(classes[int(box[0])])
        #current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
        test_image_org = cv2.rectangle(test_image_org, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        test_image_org = cv2.putText(test_image_org, label, (int(xmin), int(ymin)), 2, 1, (0,0,255), 1, cv2.LINE_AA)  
    return test_image_org

model = getCombinedModel()
model_weight_file = '/home/suraj/suraj/novus/novus_pilot_deep_learning/SSD_Segmentation/ssd_objdet_weights/ssd300_epoch-499.h5'
model.load_weights(model_weight_file, by_name=False)
"""
for layer in model.layers:
    #print(layer.name, type(layer).__name__)
    if(layer.name == 'conv2d_34' or layer.name == 'add_2'):
      print(layer.name, type(layer).__name__)
      weight = model.get_layer(layer.name).get_weights()
      print(weight)
"""
#ssd_loss = FocalLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

#model_weight_file = '/home/suraj/suraj/novus/SSD_Segmentation/weights_2/ssd_fz_fcn_epoch-14_loss-0.7813_val_loss-0.8358.h5'
#model = load_model(model_weight_file, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                               'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

path = '/home/suraj/suraj/data/SSD_RoadBoundary/images/2018-09-11-15-31-04_t_l_000090.png' #76filename7.jpg'
#test_image_org = cv2.imread(image_path)

#path = '/home/suraj/suraj/data/road_marking/suraj/mask_jsons/images/sd_frame_2331.jpg'
"""
frame = cv2.imread(path)
test_image = cv2.resize(frame, (300,300))

test_image = np.expand_dims(test_image,axis=0)
print('Input')
"""
img_org = cv2.imread(path)
# img = img[:,a:a+320]
#image = cv2.resize(img_org,(300,300))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = np.array(image, dtype = np.float32)
#print(image)
#print(image.shape)

from PIL import Image

img_obj = Image.open(path)
pil_img = np.array(img_obj)
image = cv2.resize(pil_img, (300,300))

image = np.array(image,dtype=np.float32)

#print(image)

#image[:,:,0] = image[:,:,0] - 104.0 
#image[:,:,1] = image[:,:,1] - 117.0 
#image[:,:,2] = image[:,:,2] - 123.0 
#print(image.shape)
#print(image)
image = np.expand_dims(image,axis=0)



test_image2 = image.copy()
comb_prediction = model.predict(image)
y_pred_ssd = comb_prediction
#y_pred_fcn = comb_prediction[1];print("SEGGGGG", y_pred_fcn.shape)

y_pred_fcn = 1

out_frame = display_prediction(y_pred_ssd, y_pred_fcn, img_org, confidence_threshold=0.75)

#out.write(out_frame)
#cv2.imshow('Frame',img_org)
#cv2.waitKey(0) 	


#print(model.summary())


for i in range(len(model.layers)):
    
    layer = model.layers[i]
    print(i, layer.name, layer.output.shape)
    # check for convolutional layer
    #if 'conv' not in layer.name:
    #     continue
    # summarize output shape
    #print(i, layer.name, layer.output.shape)


from keras.models import Model
new_model = Model(inputs=model.inputs, outputs=model.layers[7].output)
feature_maps = new_model.predict(test_image2)

print(feature_maps.shape)
print(feature_maps.dtype)
#print(feature_maps[:,:,:,40].shape)
print(feature_maps[0,:,:,0])
print(feature_maps[0,:,:,1])
print(feature_maps[0,:,:,2])
print(feature_maps[0,:,:,0].shape)

