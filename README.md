## 실행 방법
1. git clone https://github.com/snu-mllab/PuzzleMix.git
2. git clone https://github.com/qiaoruiyt/NoiseRobustDG.git
3. PuzzleMix/mixup.py 폴더에서 cost_matrix_dict '14' 추가하기.         # ('14': cost_matrix(14, device).unsqueeze(0))
4. python waterbirds_saliency.py --data_root [data_root] --model_dir [model_dir] --out_dir [out_dir] --save True

## TODO LIST
1. (waterbird, water)와 (landbird, land)를 random으로 뽑히게 해서 puzzle mix 하기. (현재는 처음 n개만 뽑히게 하고 있다.)
2. groupDRO로 처음부터 다시 학습하기.

![image](https://github.com/user-attachments/assets/7f51acf0-36a7-4472-ae5c-af705fe7c831)

## groupDRO로 학습하는 방법
________________________________________________________
Let, g' is the missing subgroub.\n
for t = 1,...,T:\n
  g ~ Unif(0,...,m)\n
  if g == g':\n
    x = mixup()\n
    y = 1      # The label of waterbirds is 1.\n
  else:\n
    x,y ~ P_g\n
  ...
________________________________________________________
