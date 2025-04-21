from domainbed.datasets import WILDSDataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

class WILDSWaterbirdsBG(WILDSDataset):
    """
    WILDS Waterbirds dataset wrapper for background prediction.
    """
    ENVIRONMENTS = [f"group_{i}" for i in range(6)]

    def __init__(self, root, test_envs, hparams):
        # Load base Waterbirds dataset with download enabled
        base_ds = WaterbirdsDataset(root_dir=root, download=True)
        # Override label y_array to background
        if 'background' not in base_ds.metadata_fields:
            raise ValueError("'background' 필드가 metadata_fields에 없습니다.")
        bg_idx = base_ds.metadata_fields.index('background')
        # metadata_array가 torch.Tensor일 경우 numpy로 변환 후 int로 캐스팅
        bg = base_ds.metadata_array[:, bg_idx]
        # Tensor to numpy
        if hasattr(bg, 'numpy'):
            bg = bg.numpy()
        # numpy array to int
        bg = bg.astype(int)
        # 하위 private 속성 _y_array에 덮어쓰기
        setattr(base_ds, '_y_array', bg)
        # Initialize GroupDRO dataset with group metadata
        super().__init__(base_ds, 'group', test_envs,
                         hparams['data_augmentation'], hparams)

# Bird prediction adapter: download=True를 설정합니다.
class WILDSWaterbirds(WILDSDataset):
    """
    WILDS Waterbirds dataset wrapper for bird prediction with download=True.
    """
    ENVIRONMENTS = [f"group_{i}" for i in range(6)]

    def __init__(self, root, test_envs, hparams):
        # Download dataset if not present
        base_ds = WaterbirdsDataset(root_dir=root, download=True)
        super().__init__(base_ds, 'group', test_envs,
                         hparams['data_augmentation'], hparams) 