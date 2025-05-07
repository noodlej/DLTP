import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from wilds import get_dataset

# DomainBed 알고리즘 import을 위해 NoiseRobustDG 경로 설정
sys.path.insert(0, os.path.abspath("NoiseRobustDG"))
from domainbed import algorithms
from PuzzleMix import mixup

import warnings
warnings.filterwarnings('ignore')

def load_model(model_path: str, device: torch.device):
    """저장된 모델을 불러와 Algorithm 객체로 반환합니다."""
    save_dict = torch.load(model_path, map_location=device)
    alg_name = save_dict["args"]["algorithm"]
    input_shape = save_dict["model_input_shape"]
    num_classes = save_dict["model_num_classes"]
    num_domains = save_dict["model_num_domains"]
    hparams = save_dict["model_hparams"]
    AlgClass = algorithms.get_algorithm_class(alg_name)
    model = AlgClass(input_shape, num_classes, num_domains, hparams)
    # GroupDRO buffer 'q' may mismatch shape; remove before loading
    state_dict = save_dict["model_dict"].copy()
    if "q" in state_dict:
        state_dict.pop("q")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def get_data(device, raw_ds, indices, image_count):
    meta = raw_ds.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Randomly select image_count images
    random_indices = np.random.choice(indices, image_count, replace=False)  
    images = []
    for i in range(image_count):
        img = raw_ds.get_input(random_indices[i])
        x = preprocess(img).unsqueeze(0).to(device)
        images.append(x)

    images = torch.cat(images, dim=0)

    return images

def compute_saliency(model, x, y):
    """
    x : (B,3,H,W) with requires_grad=True
    y : (B,)
    returns saliency (B,1,H,W) – L2 over RGB channels
    """
    model.zero_grad(set_to_none=True)
    logits = model.predict(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()

    grad = x.grad.detach()                    # (B,3,H,W)
    sal  = grad.pow(2).mean(1, keepdim=True).sqrt()  # (B,1,H,W)
    # normalize each map to [0,1] for nicer images
    sal_min, sal_max = sal.amin((2,3),keepdim=True), sal.amax((2,3),keepdim=True)
    sal = (sal - sal_min) / (sal_max - sal_min + 1e-8)
    return sal

def test_model(model, x, y):
    correct = 0
    total = len(y)
    with torch.no_grad():
        for i in range(len(y)):
            logits = model.predict(x[i].unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            if pred == y[i]:
                correct += 1

    print(f"Accuracy: {correct / total}")
    return correct / total

def get_saliency_map(args, device, bird_model, bg_model, mean, std):
     # Raw Waterbirds 데이터셋 로드
    raw_ds = get_dataset(dataset="waterbirds", download=False, root_dir=args.data_root)

    # 테스트 인덱스 추출
    split_arr = raw_ds.split_array
    if hasattr(split_arr, "numpy"):
        split_arr = split_arr.numpy()
    test_idx = np.where(split_arr == raw_ds.split_dict["test"])[0]

    # 메타데이터 추출 (y, background)
    meta = raw_ds.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()
    y_all = meta[:, raw_ds.metadata_fields.index("y")].astype(int)
    bg_all = meta[:, raw_ds.metadata_fields.index("background")].astype(int)

    # 서브그룹 정의
    subgroups = {
        "landbird_land": (0, 0),
        "landbird_water": (0, 1),
        "waterbird_land": (1, 0),
        "waterbird_water": (1, 1),
    }
    subgroup_indices = {}
    for name, (y_val, bg_val) in subgroups.items():
        inds = [i for i in test_idx if y_all[i] == y_val and bg_all[i] == bg_val]
        subgroup_indices[name] = inds

    # Get data
    waterbird_water = get_data(device, raw_ds, subgroup_indices["waterbird_water"], args.image_count).requires_grad_()
    landbird_water = get_data(device, raw_ds, subgroup_indices["landbird_water"], args.image_count).requires_grad_()
    waterbird_land = get_data(device, raw_ds, subgroup_indices["waterbird_land"], args.image_count).requires_grad_()
    landbird_land = get_data(device, raw_ds, subgroup_indices["landbird_land"], args.image_count).requires_grad_()
    x = torch.cat([waterbird_water, landbird_land], dim=0)
    waterbird = torch.tensor([1 for _ in range(args.image_count)]).to(device)
    land = torch.tensor([0 for _ in range(args.image_count)]).to(device)
    water = torch.tensor([1 for _ in range(args.image_count)]).to(device)
    y = torch.cat([waterbird, land], dim=0)

    # Saliency 계산
    saliency_bird = compute_saliency(bird_model, waterbird_water, waterbird)
    saliency_bg = compute_saliency(bg_model, landbird_land, water)
    sal = torch.cat([saliency_bird, saliency_bg], dim=0)
    sal = sal.squeeze(1)            # (B,1,H,W) -> (B,H,W)

    # 결과 저장
    # if args.save == True:
    #     x_vis = x.detach() * std + mean
    #     idx = 1
    #     for i in range(x.size(0)):
    #         save_image(x_vis[i],  f"{args.out_dir}/{idx:05d}_raw.png")
    #         save_image(sal[i],    f"{args.out_dir}/{idx:05d}_saliency.png")
    #         idx += 1
    #     # for(i)

    return x, sal
#END of get_saliency_map()

def save(x_puzzle, args, mean, std, start=1):
    if args.save == True:
        x_puzzle_vis = x_puzzle.detach() * std + mean
        idx = start
        for i in range(x_puzzle.size(0)):
            save_image(x_puzzle_vis[i], f"{args.out_dir}/{idx:05d}_puzzle.png")
            idx += 1
# END of save_image()

def get_puzzle_mix(args, device):
    '''
    args:
        args.model_dir: 모델 디렉토리
        args.image_count: 이미지 개수
        args.out_dir: 결과 저장 디렉토리
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    bird_model = load_model(os.path.join(args.model_dir, "model_bird.pkl"), device)
    bg_model   = load_model(os.path.join(args.model_dir, "model_background.pkl"), device)

    # Saliency Map 계산
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3,1,1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3,1,1)
    x, sal = get_saliency_map(args, device, bird_model, bg_model, mean, std)

    # Puzzle Mix 계산
    indices = torch.tensor([args.image_count * 2 - i - 1 for i in range(args.image_count * 2)]).to(device)
    x_puzzle, ratio = mixup.mixup_graph(input1=x, grad1=sal, indices=indices, block_num=8, mean=mean, std=std, transport=True, device=device)

    return x_puzzle, ratio
# END of get_puzzle_mix()

def get_puzzle_mix_reverse(args, device):
    '''
    args:
        args.model_dir: 모델 디렉토리
        args.image_count: 이미지 개수
        args.out_dir: 결과 저장 디렉토리
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    bird_model = load_model(os.path.join(args.model_dir, "model_bird.pkl"), device)
    bg_model   = load_model(os.path.join(args.model_dir, "model_background.pkl"), device)

    # Saliency Map 계산
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3,1,1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3,1,1)
    x, sal = get_saliency_map(args, device, bg_model, bird_model, mean, std)

    # Puzzle Mix 계산
    indices = torch.tensor([args.image_count * 2 - i - 1 for i in range(args.image_count * 2)]).to(device)
    x_puzzle, ratio = mixup.mixup_graph(input1=x, grad1=sal, indices=indices, block_num=8, mean=mean, std=std, transport=True, device=device)

    return x_puzzle, ratio
# END of get_puzzle_mix()

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
        "--model_dir", type=str,
        default="/data2/yws_ids25/data/AIDA",
        help="model_bird.pkl 및 model_background.pkl이 있는 디렉토리"
    )
    parser.add_argument(
        "--image_count", type=int,
        default=5,
        help="Puzzle Mix할 이미지 개수"
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="puzzled_data",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--save", type=bool,
        default=False,
        help="Saliency Map 저장 여부"
    )
    args = parser.parse_args()
    # ------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    bird_model = load_model(os.path.join(args.model_dir, "model_bird.pkl"), device)
    bg_model   = load_model(os.path.join(args.model_dir, "model_background.pkl"), device)

    # Saliency Map 계산
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3,1,1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3,1,1)
    x, sal = get_saliency_map(args, device, bird_model, bg_model, mean, std)
    
    # print(sal.shape) (args.image_count * 2, 1, 224, 224)
    # print(x.shape) (args.image_count * 2, 3, 224, 224)

    indices = torch.tensor([args.image_count * 2 - i - 1 for i in range(args.image_count * 2)]).to(device)
    x_puzzle, ratio = mixup.mixup_graph(input1=x, grad1=sal, indices=indices, block_num=8, mean=mean, std=std, transport=True, device=device)

    save(x_puzzle, args, mean, std)
#END of main()

if __name__ == "__main__":
    main()
# END of main()