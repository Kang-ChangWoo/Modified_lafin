# ModifiedLaFIn: Generative Landmark Guided Face Inpainting
- 본 저장소는 UNIST DGMS 강의 간 진행한 프로젝트에서, 성능 비교를 위해서 재현한 선행연구입니다. 
- 코드 및 논문은 "[LaFIn: Generative Landmark Guided Face Inpainting](https://arxiv.org/abs/1908.03852)" [PRCV](https://link.springer.com/chapter/10.1007/978-3-030-60633-6_2) 를 참조합니다.
- 원 문서에 대한 설명은 [`origin_repo/README.md`](origin_repo/README.md) 파일에 저장되어 있다.



### 예시 결과물 (Inpainting results)
- 참고를 위한 인페인팅 결과물은 아래와 같습니다.

![image](lafin.png)

### 필요한 라이브러리 (Prerequisites)
1. Python 3.7
2. Pytorch 1.0
3. NVIDIA GPU + CUDA cuDNN

### 설치법 (Installation)
1. 본 저장소를 클론한다.
```
git clone https://github.com/Kang-ChangWoo/Modified_lafin.git
cd lafin-master
```
* Install Pytorch
* Install python requirements:
```
pip install -r requirements.txt
```

### 재현 (implementation)
**1. 데이터셋 다운로드하기**


원 논문에서는 1)이미지 인페인팅 부분과 2) 증강된 랜드마크 검출 파트가 나눠져 있지만, 본 과정에서는 이미지 인페인팅만 실행하고자 한다. 아래의 데이터셋을 다운로드 받아야 한다.
1. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)

다운로드를 받은 이후엔, train, test, validation 을 각각 나눠서 설정해줘야 하며, [`scripts/flist.py`](scripts/flist.py)를 실행하여 관련된 파일 목록을 생성해줘야 한다.
예를 들어 celebA 데이터셋의 파일목록을 생성하고 싶다면 아래를 실행시켜야 한다.
```
mkdir datasets
python3 ./scripts/flist.py --path path_to_celebA_train_set --output ./datasets/celeba_train_images.flist
```

CelebA-HQ 데이터셋의 경우도 마찬가지다.  다만, 본 연구에서는 256x256 사이즈의 이미지를 사용하고 기존 이미지에서 센터를 자른 이후에 리사이징을 해서 학습을 진행한다..


**2. 불규칙 혹은 랜덤 마스크 생성하기**


본 모델에서는 학습을 위해서 랜덤하게 생성된 블록 마스크와 불규칙적 마스크를 조합적으로 사용한다.  불규칙적 마스크 데이터는 [Liu et al.](https://arxiv.org/abs/1804.07723)를 참고하여 활용한다.  해당 데이터셋은 [their website](http://masc.cs.gmu.edu/wiki/partialconv)에서 확인이 가능하다.


원하는 마스크 이미지를 생성한 다음엔, [`scripts/flist.py`](scripts/flist.py)를 사용해서 마스크 파일 목록을 생성해야 한다.



Getting Started
--------------------------
To use the pre-trained models, download them from the following links then copy them to corresponding checkpoints folder, like `./checkkpoints/celeba` or `./checkpoints/celeba-hq`.

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
