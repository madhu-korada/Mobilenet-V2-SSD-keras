import sys
sys.path.append('/home/suraj/suraj/novus/novus_pilot_deep_learning/ssd_keras_updated/')
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,TerminateOnNaN, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
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
from keras.utils.training_utils import multi_gpu_model
import os
import keras
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
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
    #y_pred_fcn = cv2.resize(y_pred_fcn, (test_image_org.shape[1], test_image_org.shape[0]))
    
    #print('resize faturemap', y_pred_fcn.shape)
    #print(test_image_org[..., 1].shape)
    #print(y_pred_fcn)
    #test_image_org[..., 1][y_pred_fcn > 0.5] = 255
    
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
print('Loading weights from - /home/suraj/suraj/ssd300_epoch-499.h5')

p ='/home/suraj/suraj/novus/novus_pilot_deep_learning/SSD_Segmentation/ssd_objdet_weights/ssd300_epoch-499.h5'

#from keras.models import load_model

#model = load_model(p)
model.load_weights(p, by_name=False)

print(model.summary())

#image_path = '/home/suraj/suraj/data/SSD_RoadBoundary/images/2018-09-11-15-15-28_t_l_001380.png'
#test_image_org = cv2.imread(image_path)

input_vid_path = '/home/suraj/suraj/test.mp4'
output_vid_path = '/home/suraj/suraj/test_out.mp4'

cap = cv2.VideoCapture(input_vid_path)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print('writing video') 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter(output_vid_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
# Check if camera opened successfully

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if (ret == True):

  	test_image = cv2.resize(frame, (300,300))
  	test_image = np.expand_dims(test_image,axis=0)
  	y_pred_ssd = model.predict(test_image);y_pred_fcn = y_pred_ssd
  	#y_pred_ssd = comb_prediction[0]
  	#y_pred_fcn = comb_prediction[1]
    
  	out_frame = display_prediction(y_pred_ssd, y_pred_fcn, frame, confidence_threshold=0.5)
  	
  	#out.write(out_frame)
  	cv2.imshow('Frame',frame)
  	if cv2.waitKey(10) & 0xFF == ord('q'):
  		break
  else: 
    break
 
cap.release()
