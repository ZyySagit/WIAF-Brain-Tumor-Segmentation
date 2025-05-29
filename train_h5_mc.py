# pip install nibabel scipy monai tqdm einops
import os
import torch
from monai.data import Dataset, DataLoader
import monai.transforms as transforms
from monai.losses import DiceLoss,DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.utils.enums import MetricReduction
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import random_split
from tqdm import tqdm
from monai.data import CacheDataset,PersistentDataset,SmartCacheDataset
from monai.transforms import apply_transform
import h5py


import random
import numpy as np
from torch.backends import cudnn
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
set_seed()


# 1. Dataset Preparation
data_dir = os.path.abspath("./MICCAI_BraTS2020_TrainingData") # ******Replace this with the path to the specific dataset folder.
subjects = sorted([
    os.path.join(data_dir, name) 
    for name in os.listdir(data_dir) 
    if os.path.isdir(os.path.join(data_dir, name))
])
# Check if the *.nii file exists
def check_files_exist(subj):
    subject_id = os.path.basename(subj)
    required_files = [
        f"{subject_id}_t1.nii",
        f"{subject_id}_t1ce.nii",
        f"{subject_id}_t2.nii",
        f"{subject_id}_flair.nii",
        f"{subject_id}_seg.nii",
    ]
    return all(os.path.exists(os.path.join(subj, f)) for f in required_files)
# Filter out samples of missing files
data_dicts = []
for subj in subjects:
    if check_files_exist(subj):
        subject_id = os.path.basename(subj)
        data_dicts.append({
            "t1": os.path.join(subj, f"{subject_id}_t1.nii"),
            "t1ce": os.path.join(subj, f"{subject_id}_t1ce.nii"),
            "t2": os.path.join(subj, f"{subject_id}_t2.nii"),
            "flair": os.path.join(subj, f"{subject_id}_flair.nii"),
            "label": os.path.join(subj, f"{subject_id}_seg.nii"),
        })
    else:
        print(f"Warning: skipping samples of missing files {subj}")

# 2. Divide the training validation set
val_ratio = 0.2
n_val = int(len(data_dicts) * val_ratio)
n_train = len(data_dicts) - n_val
train_data, val_data = random_split(
    data_dicts, 
    [n_train, n_val],
)

# 3. Multi-threaded preprocessing and saving as HDF5 
from multiprocessing import Pool,cpu_count
from functools import partial
def preprocess_to_hdf5(data, transform, output_path, n_chunks=cpu_count()):
    temp_template = f"{output_path}.part_{{}}.h5"
    n_samples = len(data)
    indices = np.arange(n_samples)
    chunks = np.array_split(indices, n_chunks)
    temp_files = [temp_template.format(i) for i in range(n_chunks)]
    args = list(zip(chunks, [data]*n_chunks, [transform]*n_chunks, temp_files))
    with Pool(processes=n_chunks) as pool:
        list(tqdm(pool.imap_unordered(_process_chunk, args), total=n_chunks, desc="Processing chunks"))
    with h5py.File(output_path, 'w') as main_file:
        for temp_file in tqdm(temp_files, desc="Merging files"):
            with h5py.File(temp_file, 'r') as part_file:
                for group_name in part_file.keys():
                    part_file.copy(group_name, main_file)
            os.remove(temp_file)
def _process_chunk(args):
    chunk_indices, data, transform, output_path = args
    with h5py.File(output_path, 'w') as f:
        for idx in tqdm(chunk_indices, desc=f"Chunk {output_path}"):
            item = data[idx]
            transformed = transform(item)
            grp = f.create_group(str(idx))
            for key in ['t1', 't1ce', 't2', 'flair', 'label']:
                grp.create_dataset(key, data=transformed[key].numpy(), compression='gzip')
# Creating an HDF5 dataset
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.file_path = hdf5_file
        self.transform = transform
        with h5py.File(hdf5_file, 'r') as f:
            self.indices = list(f.keys())
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            idx = self.indices[index]
            data = {
                't1': torch.from_numpy(f[idx]['t1'][:]),
                't1ce': torch.from_numpy(f[idx]['t1ce'][:]),
                't2': torch.from_numpy(f[idx]['t2'][:]),
                'flair': torch.from_numpy(f[idx]['flair'][:]),
                'label': torch.from_numpy(f[idx]['label'][:])
            }
        if self.transform:
            data = apply_transform(self.transform, data)
        return data



# 4. Data transforms
roi=(128,128,128)
trm0 = transforms.Compose([
    transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"]),
    transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
    transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    transforms.CenterSpatialCropd(keys=["t1", "t1ce", "t2", "flair", "label"], roi_size=roi),
    transforms.NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True, channel_wise=True),
])
trm = transforms.Compose([
    transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.30, spatial_axis=1),
    transforms.RandRotated(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.50, range_x=0.36, range_y=0.0, range_z=0.0),
    transforms.RandCoarseDropoutd(keys=["t1", "t1ce", "t2", "flair", "label"], holes=20, spatial_size=(-1, 7, 7), fill_value=0, prob=0.5),
    transforms.GibbsNoised(keys=["t1", "t1ce", "t2", "flair"]),
])
# ************** Preprocessing Stage Configuration (Remove comments on first run) 
preprocess_to_hdf5(train_data, trm0, "./train.h5")
preprocess_to_hdf5(val_data, trm0, "./val.h5")



def test(model):
    model.eval()
    metrics = {
        'Dice': DiceMetric(include_background=True,reduction=MetricReduction.MEAN_BATCH), 
        'DiceTC': DiceMetric(include_background=True, reduction="mean"), 
        'DiceWT': DiceMetric(include_background=True, reduction="mean"),
        'HD95': HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH),
        'HD95TC': HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean"),
        'HD95WT': HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean"),
    }
    activation = transforms.Activations(sigmoid=True) 
    threshold = transforms.AsDiscrete(threshold=0.5) 
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = apply_transform(transforms.ToDeviced(keys=["t1", "t1ce", "t2", "flair", "label"], device=device), batch)
            x = torch.cat([batch["t1"],batch["t1ce"],batch["t2"],batch["flair"]], dim=1)
            y = batch["label"]   	            # (1,3,128,128,128)
            ypred = model(x)             		    # (1,3,128,128,128)
            ypred = threshold(activation(ypred)) 	# (1,3,128,128,128)
            metrics['Dice'](y_pred=ypred, y=y)
            metrics['HD95'](y_pred=ypred, y=y)
            tc_label = torch.max(y[:,[0,2],:,:,:], dim=1).values.unsqueeze(1)		# (1,1,128,128,128)
            tc_pred = torch.max(ypred[:,[0,2],:,:,:], dim=1).values.unsqueeze(1)	# (1,1,128,128,128)
            wt_label = torch.max(y, dim=1).values.unsqueeze(1)						# (1,1,128,128,128)
            wt_pred = torch.max(ypred, dim=1).values.unsqueeze(1)					# (1,1,128,128,128)
            metrics['DiceTC'](y_pred=tc_pred, y=tc_label)
            metrics['DiceWT'](y_pred=wt_pred, y=wt_label)
            metrics['HD95TC'](y_pred=tc_pred, y=tc_label)
            metrics['HD95WT'](y_pred=wt_pred, y=wt_label)
        dice1 = metrics['Dice'].aggregate()[0].item()
        dice2 = metrics['Dice'].aggregate()[1].item()
        dice4 = metrics['Dice'].aggregate()[2].item()
        dice_tc = metrics['DiceTC'].aggregate().item()
        dice_wt = metrics['DiceWT'].aggregate().item()
        hd95_1 = metrics['HD95'].aggregate()[0].item()
        hd95_2 = metrics['HD95'].aggregate()[1].item()
        hd95_4 = metrics['HD95'].aggregate()[2].item()
        hd95_tc = metrics['HD95TC'].aggregate().item()
        hd95_wt = metrics['HD95WT'].aggregate().item()
        with open("log.txt","a") as file:
            file.write(f"\t{dice1:.4f}\t{dice2:.4f}\t{dice4:.4f}\t{dice_tc:.4f}\t{dice_wt:.4f}\t{hd95_1:.2f}\t{hd95_2:.2f}\t{hd95_4:.2f}\t{hd95_tc:.2f}\t{hd95_wt:.2f}")
        for metric in metrics.values():
            metric.reset()
    torch.save(model.state_dict(), "model_latest.pth")
    torch.cuda.empty_cache()
    return dice4+dice_tc+dice_wt

def test_final(model, test_loader, device, output_file="log_test.txt"):
    model.eval()
    activation = transforms.Activations(sigmoid=True)
    threshold = transforms.AsDiscrete(threshold=0.5)
    with open(output_file, 'w') as f:
        f.write("Sample\tDice1\tDice2\tDice4\tDiceTC\tDiceWT\tHD95_1\tHD95_2\tHD95_4\tHD95_TC\tHD95_WT\n")
        for idx, batch in enumerate(tqdm(test_loader)):
            batch = apply_transform(transforms.ToDeviced(keys=["t1", "t1ce", "t2", "flair", "label"], device=device), batch)
            x = torch.cat([batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]], dim=1)
            y = batch["label"]
            ypred = model(x)
            ypred = threshold(activation(ypred))
            dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
            hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH)
            dice_metric(y_pred=ypred, y=y)
            hd_metric(y_pred=ypred, y=y)
            dice_values = dice_metric.aggregate()
            hd_values = hd_metric.aggregate()
            dice1, dice2, dice4 = dice_values[0].item(), dice_values[1].item(), dice_values[2].item()
            hd95_1, hd95_2, hd95_4 = hd_values[0].item(), hd_values[1].item(), hd_values[2].item()
            tc_label = torch.max(y[:, [0,2], :, :, :], dim=1).values.unsqueeze(1)
            tc_pred = torch.max(ypred[:, [0,2], :, :, :], dim=1).values.unsqueeze(1)
            wt_label = torch.max(y, dim=1).values.unsqueeze(1)
            wt_pred = torch.max(ypred, dim=1).values.unsqueeze(1)
            dice_tc_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
            dice_tc_metric(y_pred=tc_pred, y=tc_label)
            dice_tc = dice_tc_metric.aggregate().item()
            hd_tc_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH)
            hd_tc_metric(y_pred=tc_pred, y=tc_label)
            hd95_tc = hd_tc_metric.aggregate().item()
            dice_wt_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
            dice_wt_metric(y_pred=wt_pred, y=wt_label)
            dice_wt = dice_wt_metric.aggregate().item()
            hd_wt_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH)
            hd_wt_metric(y_pred=wt_pred, y=wt_label)
            hd95_wt = hd_wt_metric.aggregate().item()

            f.write(f"{idx}\t{dice1:.4f}\t{dice2:.4f}\t{dice4:.4f}\t{dice_tc:.4f}\t{dice_wt:.4f}\t{hd95_1:.2f}\t{hd95_2:.2f}\t{hd95_4:.2f}\t{hd95_tc:.2f}\t{hd95_wt:.2f}\n")

            dice_metric.reset()
            hd_metric.reset()
            dice_tc_metric.reset()
            hd_tc_metric.reset()
            dice_wt_metric.reset()
            hd_wt_metric.reset()

def train(model):
    best_dice = 0.0
    patience = 15
    no_improve = 0
    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True, include_background=True, lambda_dice=0.7,lambda_ce=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    #scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=6e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)
    for epoch in range(400):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            inputs = torch.cat([batch["t1"],batch["t1ce"],batch["t2"],batch["flair"]], dim=1).to(device)
            labels = batch["label"].to(device)                  # (B,3,128,128,128)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1}\t{epoch_loss/len(train_loader):.4f}\t{lr:.4e}")
        with open("log.txt","a") as file:
            file.write(f"\n{epoch+1}\t{epoch_loss/len(train_loader):.4f}")
        if (epoch+1) % 3 == 0:
            current_dice = test(model)
            if current_dice > best_dice:
                best_dice = current_dice
                torch.save(model.state_dict(), "model_best.pth")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stop triggered on round {epoch+1}.")
                    break

# -----------------------main---------------------------------------------
from model import WIAF
model = WIAF()

params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Number of model parameters: {params:.2f}M")

train_ds = HDF5Dataset("./train.h5", transform=trm)
val_ds = HDF5Dataset("./val.h5", transform=None)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=6)#, pin_memory=True, prefetch_factor=4)
val_loader = DataLoader(val_ds, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(model)

# Load the best model after training and test it
model.load_state_dict(torch.load("model_best.pth"))
test_final(model, val_loader, device)
