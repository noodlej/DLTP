import argparse
import torch
from tqdm import tqdm

from waterbirds_saliency import get_puzzle_mix
from waterbirds_saliency import get_puzzle_mix_reverse
from waterbirds_saliency import save

def main():
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Stage-1 학습된 모델을 불러와 Puzzle Mix!"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/data2/yws_ids25/data/AIDA/data",
        help="Waterbirds 데이터의 루트 디렉토리"
    )
    parser.add_argument(
        "--puzzled_data_root", type=str,
        default="/data2/yws_ids25/data/AIDA/data/puzzled_data/train",
        help="puzzled data 저장 디렉토리"
    )
    parser.add_argument(
        "--model_dir", type=str,
        default="/data2/yws_ids25/data/AIDA",
        help="model_bird.pkl 및 model_background.pkl이 있는 디렉토리"
    )
    parser.add_argument(
        "--image_count", type=int,
        default=16,
        help="batch size / 2"
    )
    parser.add_argument(
        "--data_count", type=int,
        default=16,
        help="data 개수: batch size * data_count "
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="/data2/yws_ids25/data/AIDA/data/puzzled_data/train",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--save", type=bool,
        default=True,
        help="Puzzle Mix 이미지 저장 여부"
    )
    parser.add_argument(
        "--reverse", type=bool,
        default=False,
        help="Puzzle Mix 이미지 저장 여부"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3,1,1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3,1,1)

    for i in tqdm(range(args.data_count)):
        if args.reverse:
            puzzled_img, _ = get_puzzle_mix_reverse(args, device)
        else:
            puzzled_img, _ = get_puzzle_mix(args, device)
        save(puzzled_img, args, mean, std, start=i*args.image_count + 1)
# END of main()

if __name__ == "__main__":
    main()