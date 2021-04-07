#필요한 모듈을 import 하는 부분
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf

#absl모듈의 flags 객체를 import 시킨 후 flags 객체를 활용하여 변수 값들을 정의하여 사용한다.
#사용시에는 FLAGS.key값을 통해 사용하며 선언할 때는 flags.DEFINE_자료형(key, value, info)로 선언한다.
flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', './data/yolov4.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')

def main(_argv):
    #Yolo-tiny버전이 아닌지 if문을 통해 구분한다. 
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        #tiny버전이 아닐 경우 Yolo-v4모델을 가져오고 anchor박스의 정보도 함께 가져온다.
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)

    #클래스개수, 박스의 XYSCALE을 Yolo-v4의 cfg파일에서 불러오고 input_size와 image_path를 미리 정의한 flags객체의 size와 image값으로 정의한다.
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size
    image_path = FLAGS.image

    #cv2모듈을 통해 이미지를 불러오고 불러온 이미지를 BGR이미지를 RGB로 바꿔준다.
    #이는 컬러 사진을 opencv에서는 BGR순서로 저장하는데 matplotlib에서는 RGB로 저장하기 때문이다.
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    #이미지 데이터들을 배열로 바꿔주고 데이터타입을 float32로 변환해준다.
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    #framework가 tf로 정의된 경우 FLAGS.model이 어떻게 정의되었는지에 따라 불러오는 모델이 다르다.
    #지금의 경우는 Yolo-v4를 다루고 있으므로 FLAGS.model이 yolov4로 정의된 경우만 보겠다.
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)

            #YOLOv4에 input 레이어와 클래스를 넣어주어 feature map을 생성하고 바운딩 박스를 예측하기 위한 리스트를 선언해준다.
            #이후 반복문을 통해 예측된 바운딩박스의 좌표를 리스트에 넣어준 뒤 이것을 model에 input레이어와 함께 넣어 model을 생성해준다.
            #그 다음 미리 학습된 weights값들을 load해온다.    
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights(model, FLAGS.weights)

        model.summary()
        #이후 원래 이미지 데이터에서 예측된 바운딩 박스를 표시해준다.
        pred_bbox = model.predict(image_data)
    else:.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        
    #이후 표시된 바운딩 박스 중 유효한 바운딩 박스들만 남기는 작업을 한 후 최종적으로 pred_bbox에 저장한다.
    if FLAGS.model == 'yolov4':
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    else:
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')

    #cv2모듈을 사용하여 예측한 바운딩박스가 표시된 이미지를 출력한다.
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
