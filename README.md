## 실행 방법
1. git clone https://github.com/snu-mllab/PuzzleMix.git
2. git clone https://github.com/qiaoruiyt/NoiseRobustDG.git
3. PuzzleMix/mixup.py 폴더에서 cost_matrix_dict '14' 추가하기.         # ('14': cost_matrix(14, device).unsqueeze(0))
4. python waterbirds_saliency.py --data_root [data_root] --model_dir [model_dir] --out_dir [out_dir] --save True

## TODO LIST
1. (waterbird, water)와 (landbird, land)를 random으로 뽑히게 해서 puzzle mix 하기. (현재는 처음 n개만 뽑히게 하고 있다.)
2. groupDRO로 처음부터 다시 학습하기.

![image](https://github.com/user-attachments/assets/7f51acf0-36a7-4472-ae5c-af705fe7c831)



# AIDA Puzzle 학습 결과

## 학습 개요
- 모델: ResNet50 (ImageNet 사전학습 가중치 사용)
- 학습 단계: Stage 2
- GPU: CUDA_VISIBLE_DEVICES=2

## 학습 결과

### 최종 성능 (Epoch 9)
#### 검증 세트 (Validation)
- Group 1: 94.65%
- Group 2: 82.19%
- Group 3: 79.69%
- Group 4: 95.49%

#### 테스트 세트 (Test)
- Group 1: 94.86%
- Group 2: 80.84%
- Group 3: 81.78%
- Group 4: 92.06%

### 학습 과정
- 총 9 에포크 동안 학습 진행
- Early stopping이 적용되어 9번째 에포크에서 학습 종료
- Group 1과 Group 4는 대부분의 에포크에서 90% 이상의 높은 정확도를 보임
- Group 2와 Group 3은 학습 초기에 낮은 정확도를 보이다가 점진적으로 개선됨
