1. git clone https://github.com/snu-mllab/PuzzleMix.git
2. git clone https://github.com/qiaoruiyt/NoiseRobustDG.git
3. PuzzleMix/mixup.py 폴더에서 cost_matrix_dict '14' 추가하기.         # ('14': cost_matrix(8, device).unsqueeze(0))
4. python waterbirds_saliency.py --data_root [data_root] --model_dir [model_dir] --out_dir [out_dir] --save True
