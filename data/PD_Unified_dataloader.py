from torch.utils.data import DataLoader, ConcatDataset
from utils.utils import load_yaml
from data.PD_base_motion_dataset import PDGaMDataset as PDGaMDatasetMotion
from data.PD_base_motion_dataset import FullSequencePDGaMDataset
from data.PD_base_motion_dataset import UnifiedMotionDataset
from data.PD_base_action2motion_dataset import UnifiedAction2MotionDataset
from data.PD_base_action2motion_dataset import PDGaMDataset as PDGaMDatasetAction2Motion

def get_data_loaders(opt, base_config_path='', split='train', drop_last=True):
    datasets = get_datasets(opt, base_config_path, split)
    # base_config = load_yaml(base_config_path)
    if opt.model_stage == 'vq':
        unified_dataset = UnifiedMotionDataset(datasets)
    else:
        unified_dataset = UnifiedAction2MotionDataset(datasets)
    shuffle = (split == 'train')
    data_loader = DataLoader(
        unified_dataset, 
        batch_size=opt.batch_size, 
        drop_last=drop_last, num_workers=4,
        shuffle=shuffle, pin_memory=True, persistent_workers=True
    )
    return data_loader, unified_dataset

def get_datasets(opt, base_config_path, split):
    if opt.model_stage == 'vq':
        if opt.get_whole_motion:
            db = FullSequencePDGaMDataset
        else:
            db = PDGaMDatasetMotion
    else:
        db = PDGaMDatasetAction2Motion
    available_datasets = {
        'pdgam': db
    }
    selected_datasets = opt.dataset_name
    datasets = []
    for dataset_name in selected_datasets:
        if dataset_name in available_datasets:
            dataset_class = available_datasets[dataset_name]
            dataset = dataset_class(opt=opt, split=split)
            datasets.append(dataset)
        else:
            print(f"Dataset {dataset_name} is not available")
            SystemExit()

    return datasets

def load_config(dataset_name, base_config_path):
    base_config = load_yaml(base_config_path)
    dataset_config = load_yaml(f'./data/configs/{dataset_name.lower()}.yaml')
    merged_config = {**base_config, **dataset_config}
    return merged_config
