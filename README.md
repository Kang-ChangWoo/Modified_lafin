# ModifiedLaFIn: Generative Landmark Guided Face Inpainting
- 본 저장소는 UNIST DGMS 강의 간 진행한 프로젝트에서, 성능 비교를 위해서 재현한 선행연구입니다. 
- 코드 및 논문은 "[LaFIn: Generative Landmark Guided Face Inpainting](https://arxiv.org/abs/1908.03852)" [PRCV](https://link.springer.com/chapter/10.1007/978-3-030-60633-6_2) 를 참조합니다.
- 원 문서에 대한 설명은 [`origin_repo/README.md`](origin_repo/README.md) 파일에 저장되어 있다.



### 예시 결과물 (Inpainting results)
- 참고를 위한 인페인팅 결과물은 아래와 같습니다.

![image](lafin.png)

### 필요한 라이브러리 (Prerequisites)
( **RTX 3090-24GB** 환경에서 구동하기 위한 설정입니다. )
- **Pytorch 1.11.0** (modified from Pytorch 1.0)
- **Python 3.9.5** (modified from Python 3.7)
- **NVIDIA GPU + CUDA 11.3** (modified from NVIDIA GPU + CUDA cuDNN)

그 외 버전에 영향 없는 라이브러리는 필요 시 설치해주시면 됩니다.

- Tensorboard
- Matlab
- Pandas
- Numpy



### 설치법 (Installation)
**1. 본 저장소를 클론한다.**
```
git clone https://github.com/Kang-ChangWoo/Modified_lafin.git
cd lafin-master
```
**2. 앞서서 언급된 필요한 라이브러리를 차례로 설치해준다.**


### 재현 (implementation)
**1. 데이터셋 다운로드 및 파일목록 생성하기**


DGMS 강의 간 제공된 평가 데이터셋의 경우 600개의 이미지와 마스크로 구성되어 있고, 전처리가 필요하기 때문에 본 Git 저장소에 저장해뒀기 때문에 따로 저장할 필요가 없다. 
1. [DGSM 평가 데이터](examples/images/000.png)


필요한 데이터의 파일 경로가 다르기 때문에 [`scripts/flist.py`](scripts/flist.py)에 `path`와 

다운로드를 받은 이후엔, train, test, validation 을 각각 나눠서 설정해줘야 하며, [`scripts/flist.py`](scripts/flist.py)를 실행하여 관련된 파일 목록을 생성해줘야 한다.
예를 들어 celebA 데이터셋의 파일목록을 생성하고 싶다면 아래를 실행시켜야 한다.
```
mkdir datasets
python3 ./scripts/flist.py --path path_to_DGMS-validation_train_set --output ./datasets/DGMS-validation_train_images.flist
```



**2. 기학습된 네트워크 가중치 다운로드 받기**
선행연구에서 제공되는 네트워크 가중치를 다운로드 받아야 한다.

1. [CelebA pretrained weights](https://drive.google.com/open?id=1lGFEbxbtZwpPA9JXF-bhv12Tdi9Zt08G)
2. [CelebA-HQ pretrained weights](https://drive.google.com/open?id=1Xwljrct3k75_ModHCkwcNjJk3Fsvv-ra) 

To use the pre-trained models, download them from the following links then copy them to corresponding checkpoints folder, like `./checkkpoints/celeba` or `./checkpoints/celeba-hq`.

Getting Started
--------------------------


[CelebA](https://drive.google.com/open?id=1lGFEbxbtZwpPA9JXF-bhv12Tdi9Zt08G) | [CelebA-HQ](https://drive.google.com/open?id=1Xwljrct3k75_ModHCkwcNjJk3Fsvv-ra) | [WFLW](https://drive.google.com/open?id=1I2MzHre1U3wqTu5ZmGD36OiXPaNqlOKb)

### 0.Quick Testing
To hold a quick-testing of our inpaint model, download our pre-trained models of CelebA-HQ and put them into `checkpoints/example`, then run:
```
python3 test.py --model 3 --checkpoints ./checkpoints/example
```
and check the results in `checkpoints/example/results`.

Please notice that, as no face detector is applied at the landmark prediction stage, the landmark predictor is sensitive to the scale of face images. If you find the provided pre-trained model generalizes poorly on your own dataset, you may need to train your own model basing on your dataset.

### 1.Image Inpaint Part
#### 1) Training 
To train the model, create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoint folder. Following comments on `config.yml.example` to set `config.yml`.

The inpaint model is trained in two stages: 1) train the landmark prediction model, 2) train the image inpaint model. To train the model, run:

```
python train.py --model [stage] --checkpoints [path to checkpoints]
``` 

For example, to train the landmark prediction model on CelebA dataset, the checkpoints folder is `./checkpoints/celeba` folder, run:

```
python3 train.py --model 1 --checkpoints ./checkpoints/celeba
```

The number of training iterations can be changed by setting `MAX_ITERS` in `config.yml`.

#### 2) Testing
To test the model, create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoint folder. Following comments on `config.yml.example` to set `config.yml`.


The model can be tested in 3 stages (landmark prediction model, inpaint model(inpaint using ground-truth landmarks) and joint model(inpainting using predicted landmarks)).
The file list of test images and landmarks can be generated using `scripts/flist.py` then set in the `config.yml` file. For testing stage 3, the test landmark file list is not needed.

For example, to test the inpaint model on CelebA dataset under `./checkpoints/celeba` folder, run:
```
python3 test.py --model 2 --checkpoints ./checkpoints/celeba
```
### 2.Augmented Landmark Detection Part
#### 1) Training
We suppose you use WFLW dataset to validate the augmented landmark detection method.
To validate the augmentation methods, a landmark-guided inpaint model trained on WFLW (stage 2) is needed. You can train it by yourself following above steps or use the pre-trained models.

Create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoint folder. Following comments on `config.yml.example` to set `config.yml`.
Remeber set `AUGMENTATION_TRAIN = 1` to enable augmentation with inpainted images, amd `LANDMARK_POINTS = 98` in `config.yml`.
Then run:
```
python3 train.py --model 1 --checkpoints ./checkpoints/wflw
```
to start augmentated training.

#### 2) Testing
Create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoints folder. Following comments on `config.yml.example` to set `config.yml`.
Then run:
```
python3 test.py --model 1 --checkpoints ./checkpoints/wflw
```
to start testing the landmark detection model on WFLW. Set `MASK = 0` in `config.yml` to achieve the highest accuracy.
