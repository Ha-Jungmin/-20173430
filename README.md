# -20173430
차량지능 기초 과제

1. Open Dataset

1) waymo_open_dataset

데이터 설명)
Waymo Open Dataset은 다양한 조건에서 Waymo 자율 주행 자동차가 수집 한 고해상도 센서 데이터로 구성됩니다. 이 데이터는 비상업적 용도로 사용이 허가되었습니다.

홈페이지)
http://www.waymo.com/open/ (라이센스를 취득하여 데이터를 얻기 위해선 홈페이지에 접속하여 라이센스를 신청하여야한다.)

데이터 사용을 위한 import 코드)

import tensorflow_datasets as tfds

tfds.object_detection.WaymoOpenDataset

버전)
v1.2(기본 구성)

데이터 크기)
336.62GB

Feature)
카메라 전면부에서 촬영한 영상, 전면부의 영상 중 오른쪽과 왼쪽의 영상, 자동차의 좌측과 우측에서 촬영한 영상을 제공하며 input shape은 전면부의 영상은 전부 (1280,1920,3)이고 좌측과 우측의 영상은 (886, 1920, 3)이다. 
데이터를 사용할 때 불러올 수 있는 key값들로는
camera_FRONT, camera_FRONT_LEFT, camera_FRONT_RIGHT, camera_SIDE_LEFT, camera_SIDE_RIGHT가 있고 그 안에 image, labels, bbox, type과 같은 key값들이 들어있고 박스의 이름을 나타내는 context가 있다. 

데이터 구성)
 ‘train’, ‘validation’으로 구성 되어있음. 또한, split을 통해 데이터를 나눌 수 있다. train data가 158,081개, validation data가 39,987개로 구성 되어있다.

만약 80%의 train data를 사용하고 싶다면 
ex) train_data = tfds.load(name=’waymo_open_dataset’, split=’train[:80%]’)
이렇게 사용이 가능하다.

![1](https://user-images.githubusercontent.com/76420366/113827702-925bc400-97be-11eb-9cd2-523c172265f1.png)

< camera_FRONT의 이미지와 레이블링된 바운딩 박스 >

![2](https://user-images.githubusercontent.com/76420366/113827740-9b4c9580-97be-11eb-84bb-b10791ecb575.png)

< camera_FRONT_LEFT의 이미지와 레이블링된 바운딩 박스 >

![3](https://user-images.githubusercontent.com/76420366/113827748-9c7dc280-97be-11eb-85cf-9ed38c7d2397.png)

< camera_FRONT_RIGHT의 이미지와 레이블링된 바운딩 박스 >

![4](https://user-images.githubusercontent.com/76420366/113827751-9d165900-97be-11eb-90be-b5a6f1577500.png)

< camera_SIdE_LEFT의 이미지와 레이블링된 바운딩 박스 >

![5](https://user-images.githubusercontent.com/76420366/113827757-9e478600-97be-11eb-98d5-4781b909563e.png)

< camera_SIDE_RIGHT의 이미지와 레이블링된 바운딩 박스 >

활용 예)
 자율주행에 있어 보행자와 자동차를 인식하는 것은 중요한 일이다. 
따라서 이 데이터셋은 주행 중 보행자와 자동차를 인식하는데 사용할 수 있다.


2) GTSRB dataset

데이터 설명)
 2011년에 독일에서 만들어진 표지판 이미지 데이터셋이다, 우리가 알아볼 데이터는 GTSRB_Final_Training_Images.zip 파일의 데이터이다. 데이터를 활용하기 전 이미지 resize, one_hot_encoding 등의 전처리가 필요하다.

용량)
276.29GB

Feature)
 (unknown, unknown, 3) : 3채널의 다양한 크기의 이미지셋, 데이터를 활용할 때는 resize를 통해 이미지 크기를 조절해줘야함

홈페이지)
https://benchmark.ini.rub.de/gtsrb_dataset.html

다운로드 링크)
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

데이터 구성)
 39209개의 이미지와 레이블 데이터, 아래와 같은 이미지들과 그것들에 대한 레이블이 된 바운딩박스의 좌표가 데이터셋에 들어있다.

![6](https://user-images.githubusercontent.com/76420366/113827762-9f78b300-97be-11eb-8c29-a8cf5f050f09.png)

 추가 사항:　

결과)
 하나의 csv파일로 저장이 되며 csv 파일안에는 두 개의 열로 나눠서 저장이 된다.
첫 번째 열은 파일의 이름이 저장되고 두 번째 열은 클래스 이름이 저장된다.

전처리 연산)
1) HOG(Histogram of Oriented Gradients) features
일반적으로 보행자 검출이나 사람의 형태를 검출하는데 사용하는 방법이다.

2) Haar-like features
특징점을 추출하는데 사용되는 연산이다.

3) Hue Histograms
이미지에서 Hue값을 가지고 히스토그램화 하는 연산이다.

활용 예)
 자율주행에 있어 표지판인지는 중요한 요소 중 하나이다. 표지판을 인지해야 다음에 자동차가 수행해야할 미션이 주어지기 때문이다. 따라서 이 데이터셋을 활용하여 주행 중 표지판을 인지할 수 있도록 학습시키는 방향으로 활용이 가능하다.

3) BDD100k

데이터 설명)
 버클리 인공지능 연구 실험실(BAIR) 에서 40초의 비디오 시퀀스, 720 픽셀 해상도, 초당 30프레임의 고화질로 취득된 10만개 비디오 시퀀스 데이터셋을 오픈소스로 공개했습니다. 거친 주행 환경 구현, GPS 정보, IMU 데이터, 타임 스탬프가 포함되어 있다. 녹화된 비디오는 비오는 날씨, 흐린 날씨, 맑은 날씨, 안개와 같은 다양한 날씨 조건이 기록되어 있다. 데이터 세트는 낮과 밤이 적절한 비율로 기록되어 있다. 이 비디오는 5만회의 운행을 통해 수집했다. 다양한 기상조건에서 기록되었다.
 
![7](https://user-images.githubusercontent.com/76420366/113827795-a8698480-97be-11eb-8fa8-5e1c9db213dd.png)

< 데이터 예시 >

데이터 구성)　
 라벨링 된 훈련용 데이터셋 – 1,941,237개
주행영역 세그먼테이션 – 70,000개의 훈련용, 10,000개의 검증용 픽셀 맵도 제공이 된다.
다양한 날씨, 다양한 도로환경, 낮과 밤 등 많은 타입의 데이터들이 포함됨
각 데이터를 카테고리별로 나눴을 경우 데이터의 개수를 그래프화하면 아래의 그래프와 같이 나온다.

![8](https://user-images.githubusercontent.com/76420366/113827797-a9021b00-97be-11eb-93df-a2669c81d5aa.png)

![9](https://user-images.githubusercontent.com/76420366/113827798-a9021b00-97be-11eb-878f-236c24bca40d.png)


Feature)
 10만장의 이미지 주석 파일은 라벨된 객체의 리스트를 가지고 있습니다. 각 객체는 source image URL, 카테고리 라벨, 사이즈(시작 좌표, 끝 좌표, 너비, 높이), truncation & occlusion의 특성, 신호등 색을 포함합니다.

추가 사항:

 레이블링 중 적색의 차선은 현재 주행 가능한 경로 즉, 현재 주행 중인 경로이고 파란색은 바꿀 수 있는 차선을 파란색으로 표시한다.

활용예)
 도로, 자동차, 보행자, 표지판, 신호등등 자율주행에 필요한 거의 모든 객체를 학습할 수 있도록 데이터가 구성되어 있고 데이터가 학습에 알맞게 구성되어 있어 특별한 전처리 없이도 학습이 용이하다는 장점이 있다.

2. 자율주행 인지관련 Open Source

1) YOLO-v4

open source 설명)
- 실시간으로 객체를 탐지할 수 있도록 고안된 네트워크
- 헤드 부분에서 클래스를 예측하는 classification과 바운딩 박스의 좌표를 찾는 regression이 동시에 일어난다.

YOLOv4의 아키텍쳐)
backbone: CSP-Darknet53
Neck : SPP, PAN
Head : YOLO-v3 

![10](https://user-images.githubusercontent.com/76420366/113827801-a99ab180-97be-11eb-952f-1a5eb462b78b.png)

 baselayer를 X_0’와 X_0”로 나누어 X_0”를 합성곱한 후 X_0”레이어와 합치는 연산을 하여 하나의 Dense Layer를 구성하고 이를 k번 한 후 Partial Dense Block이 끝나는 지점에서 X_0’와 합친 후 X_T를 만들고 Partial Transition Layer에서 합성곱을 하여 X_U를 output으로 출력한다. 이를 CSP 구조라고 한다. CSP-Darknet53은 레이어의 개수가 53개인 CSP구조의 darknet구조라는 뜻이다.
 
 또한, input shape은 (512, 512, 3)로 작은 객체까지 검출 가능하게 하였고, 레이어의 수를 이전 버전보다 늘려 receptive field를 물리적으로 늘려주었다. 또한, 파라미터의 수를 늘려 다양한 객체들을 동시에 많이 탐지할 수 있도록 해주었다.
 
 ![11](https://user-images.githubusercontent.com/76420366/113827803-a99ab180-97be-11eb-96fd-2a57acd28e86.png)
 
 모델에대해 설명하면 CSP-Darknet53을 이용하여 특징을 추출한 후 SPP와 PAN을 사용하여 feature의 개수를 줄이며 마지막으로 YOLOv3에서 사용했던 방식과 같이 객체의 위치를 regression을 통해 찾아낸다. 그 결과 아래의 사진과 같이 감지된 객체들이 바운딩 박스를 통해 확률로 나타내진다.

![12](https://user-images.githubusercontent.com/76420366/113827804-aa334800-97be-11eb-8e34-6f9707f2fbff.png)

2) fast-RCNN

open source 설명)
- R-CNN이라는 모델을 기반으로 발전한 객체 탐지 모델이다.
- 한번에 객체의 classification과 바운딩 박스 regression이 한번에 진행되는 single-stage로 학습이 진행된다.

순전파 학습

![13](https://user-images.githubusercontent.com/76420366/113827783-a6072a80-97be-11eb-849a-4045f6952351.png)

fast-RCNN에서 순전파 학습이 일어나는 과정은 다음과 같다.

(1) 먼저 input image에서 미리 ROI를 추출한다.

(2) 이후 input image 전체에 대해 합성곱을 이용하여 feature map을 생성한다.

(3) 생성된 feature map에 처음 추출한 ROI를 적용시켜 pooling layer를 통해 이미지 크기를 늘려준다.

(4) 이후 dense layer를 통과 시키며 2개의 branch로 나눈다.

(5) 하나는 객체가 무엇인지 분류하는 classification에 사용하고 다른 하나는 바운딩 박스의 좌표를 예측하는 regression에 사용을 한다.

(6) 이후 분류 손실값과 회귀 손실값을 더해주어 클래스도 일치하고 좌표값도 일치하는 모델로 학습을 진행하면 된다.

역전파 학습

![14](https://user-images.githubusercontent.com/76420366/113827784-a7385780-97be-11eb-87c8-020d33a54253.png)

특징을 추출하는 부분은 기존에 학습된 CNN모델인 VGG16모델이나 ResNet50모델을 사용하되 최종단의 분류 레이어는 떼어내고 사용을 해야한다. 예를 들어 최종단의 softmax가 1000개의 클래스를 분류하는 모델이라면 이 부분을 우리가 사용할 모델이 분류해야할 클래스의 개수로 변형하여 학습을 시켜주어 학습이 일어나도록 해줘야한다. 

그 결과 아래의 사진과 같이 감지된 객체들이 바운딩 박스를 통해 나타내진다.

![15](https://user-images.githubusercontent.com/76420366/113827791-a7d0ee00-97be-11eb-8976-ef69cde3e222.png)

3. 구현 환경 및 실행 관련 코드

1) 구현 환경
google colab - (gpu:telsa k80)

2) 실행 관련 코드

(1) git clone을 통해 Yolo-v4를 colab 클라우드 컴퓨터에 다운 받는다. 

!git clone https://github.com/AlexeyAB/darknet
 
(2) 다운 받은 모델의 구동 환경을 바꿔준다. s/는 치환을 타나내며 Makefile 내의 opencv,      GPU, CUDNN, CUDNN_HALF를 모두 1로 바꿔 사용해준다.

%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

(3) 마지막으로 make 명령을 통해 바뀐 옵션으로 빌드를 진행한다.

!make

(4) Yolo-v4개발자가 제공하는 미리 학습된 weights를 다운 받는다.

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

(5) 학습된 이미지를 matplotlib를 이용하여 시각화할 수 있도록 imshow 함수를 만들어주
    고 colab에 사진 및 영상을 업로드하여 학습된 Yolo-v4를 테스트 할 수 있도록 upload
    와 download함수를 만든다.

(6) Yolo-v4에서 제공되는 test 이미지에서 객체를 탐지하여 학습이 잘 되었는지 확인한
    다. (darknet을 실행시키고 그 뒤로 detector test를 실행시키고 뒤로 클래스명, 모델설
    정, 가중치, 테스트할 파일을 넣어준다.)
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg

(7) imshow를 통해 학습된 이미지를 시각화하여 바운딩 박스가 잘 그려졌는지 확인한다.

imshow(‘prediction.jpg’)

+) 추가 사항
colab에 영상을 업로드 하여 학습된 모델을 영상에 적용시켜 테스트한 영상을 다운로드하기

(1) 우선 위의 (5)번 사항까지 진행을 한 후 upload()를 통해 테스트할 영상을 업로드한다.

upload()

(2) 이후 detector demo를 통해 학습된 모델을 테스트 영상에 적용시킨다.

!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show 영상이름.mp4 -i 0 -out_filename Output영상이름.avi

(3) 학습 도중에는 영상이 보이지 않으므로 download()를 통해 Yolo-v4를 통해 객체를 인식하는 영상을 다운로드한다.

download(‘Output영상이름.avi’)
