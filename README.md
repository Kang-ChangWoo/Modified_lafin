# ModifiedLaFIn: Generative Landmark Guided Face Inpainting
- 본 저장소는 UNIST DGMS 강의 간 진행한 프로젝트에서, 성능 비교를 위해서 재현한 선행연구입니다. 
- 코드 및 논문은 "[LaFIn: Generative Landmark Guided Face Inpainting](https://arxiv.org/abs/1908.03852)" [PRCV](https://link.springer.com/chapter/10.1007/978-3-030-60633-6_2) 를 참조합니다.
- 원 문서에 대한 설명은 [`origin_repo/README.md`](origin_repo/README.md) 파일에 저장되어 있습니다.



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
**1. 본 저장소를 클론합니다.**
```
git clone https://github.com/Kang-ChangWoo/Modified_lafin.git
cd lafin-master
```
**2. 앞서서 언급된 필요한 라이브러리를 차례로 설치한다.**


### 재현 (implementation)
**1. 데이터셋 다운로드 및 파일목록 생성하기**


DGMS 강의 간 제공된 평가 데이터셋의 경우 600개의 이미지와 마스크로 구성되어 있습니다.  하지만 모델을 테스트하기 위해선 전처리가 다소 필요합니다.  때문에 본 저장소에 미리 컴퓨팅된 데이터를 업로드해놨기 때문에 따로 데이터를 다운받을 필요는 없습니다.
- [DGSM 평가 데이터](examples/images/000.png)


다운로드를 받은 이후엔, train, test, validation 을 각각 나눠서 설정해야 합니다. 또한 필요한 데이터의 파일 경로가 다르기 때문에, [`scripts/flist.py`](scripts/flist.py)를 실행하여 관련된 파일 목록을 생성해줘야 합니다.  예를 들어 celebA 데이터셋의 파일목록을 생성하고 싶다면 아래를 실행시켜야 합니다.

```
mkdir datasets
python3 ./scripts/flist.py --path path_to_DGMS-validation_train_set --output ./datasets/DGMS-validation_train_images.flist
```



**2. 기학습된 네트워크 가중치 다운로드 받기**
선행연구에서 제공되는 네트워크 가중치를 다운로드 받아야 합니다.

- [CelebA pretrained weights](https://drive.google.com/open?id=1lGFEbxbtZwpPA9JXF-bhv12Tdi9Zt08G)
- [CelebA-HQ pretrained weights](https://drive.google.com/open?id=1Xwljrct3k75_ModHCkwcNjJk3Fsvv-ra) 

기학습된 네트워크 가중치를 사용하기 위해서, 해당 데이터를 `./testing/` 위치에 저장해야 합니다.



**3. DSGM 데이터셋 테스트하기**

빠르게 테스트하기 위해서 아래 코드를 그대로 실행시켜주면 됩니다.
```
python3 test.py --model 3 --checkpoints ./testing
```
해당하는 결과는 `checkpoints/testing/results`에서 참고하실 수 있습니다.
