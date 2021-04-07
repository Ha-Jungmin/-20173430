from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

#텐서플로우 사용시 gpu메모리 할당을 증가시키는 코드
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

#OptionParser()를 사용하여 파이썬 실행시 옵션에 따라 다르게 동작할 수 있도록 옵션들을 넣어준다.
parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

#feature map을 생성할 때 사용할 CNN구조를 resnet50과 vgg 둘 중 하나로 선택할 수 있도록 한다.
if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

#이미지를 resize하여 resize된 이미지와 비율을 return 값으로 내보낸다.
def format_img_size(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

#이미지 스케일링을 진행하고 채널을 초기화하여 이미지를 반환시켜준다.
def format_img_channels(img, C):
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

#위에서 채널이 초기화된 이미지와 이미지 비율을 이용하여 input으로 들어온 이미지를 변환하여 반환시켜준다.
def format_img(img, C):
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

#원래 갖고 있던 바운딩 박스의 좌표값들을 비율이 변환된 이미지에 맞게 좌표들도 변환시켜 (real_x1, real_y1, real_x2 ,real_y2)로 변환시켜준다.
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

#클래스를 mapping 시켜주고 feature에서 계산할 ROI의 개수를 options에서 불러온다.
class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

#CNN가 resnet50일 경우 feature의 크기가 1024, vgg일 경우 feature의 크기가 512로 설정한다.
if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.common.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

#input image shape, roi input shape, feature map input shape을 초기화해준다.
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

#feature map을 생성하는 layer를 정의해준다.(fast-RCNN의 기본 CNN은 resnet50이다.)
shared_layers = nn.nn_base(img_input, trainable=True)

#anchor박스 수, rpn레이어, 분류기를 정의한다. rpn 레이어는 CNN을 통해 얻은 feature map에서 ROI를 계산하는데 사용한다.
#이는 이후 classifier에 전달되어 객체 탐지에 활용한다.
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

#rpn모델과 분류기만 있는 모델을 정의한다.
#이 때 rpn모델에는 이미지와 rpn 레이어가 들어가고 분류기에는 feature map과 rpn을 통해 계산된 roi와 위에서 정의한 classifier가 들어간다.
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print(f'Loading weights from {C.model_path}')

# rpn과 분류기에 사용될 weights값들을 load하고 이후 optimizer와 loss함수를 지정하여 컴파일해준다.
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

#이미지를 읽어오고 이후 이미지들을 일정한 비율로 resize시키고 채널을 초기화하는 등의 전처리를 한다.
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)

	if K.common.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	#rpn으로부터 feature map에서의 ROI를 계산하고 계산된 ROI 중 일치 비율이 0.7이상인 ROI만 남긴다.
	[Y1, Y2, F] = model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

	#x1,y1,x2,y2로 되어있던 좌표를 x,y,w,h로 바꿔준다.
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	
	bboxes = {}
	probs = {}

        #fast-RCNN의 동작 중 원래 feature map에서 rpn을 통해 얻은 ROI로 ROI pooling이 진행되는 과정이다.
	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                #이후 ROI pooling을 적용한 후 다시 분류기를 통해 객체 탐지를 한다.
		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

        #예측된 바운딩 박스를 그리고 겹치는 바운딩 박스들을 일정 임계값을 통해 없애나가며 최종적으로 객체를 탐지한 이미지를 보여준다.
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = f'{key}: {int(100*new_probs[jk])}'
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	print(f'Elapsed time = {time.time() - st)}'
	print(all_dets)

	
	cv2.imwrite('./results_imgs-fp-mappen-test/{}.png'.format(os.path.splitext(str(img_name))[0]),img)
