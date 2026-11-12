# ==================== CELL 0 ====================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import json

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
class ChannelAttention(nn.Module):
    """Channel Attention - focuses on 'what' is meaningful"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Spatial Attention - focuses on 'where' is meaningful"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class SkipConnectionAttention(nn.Module):
    """Attention for skip connections"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(self.cbam(x))
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """ASPP with attention"""
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18]):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for dilation in dilations:
            modules.append(ASPPConv(in_channels, out_channels, dilation))
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.attention = CBAM(out_channels, reduction=16)
    
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        x = self.project(res)
        return self.attention(x)

class Decoder(nn.Module):
    """Decoder with skip connections and attention"""
    def __init__(self, low_level_channels, num_classes):
        super().__init__()
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.skip_attention = SkipConnectionAttention(48, reduction=4)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256, reduction=16),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x, low_level_feat):
        low_level_feat = self.skip_attention(self.low_level_conv(low_level_feat))
        x = F.interpolate(x, size=low_level_feat.shape[-2:], 
                         mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        return self.decoder(x)
class FoodSegmentationModel(nn.Module):
    """Complete segmentation model with attention everywhere"""
    def __init__(self, num_classes=104, pretrained_backbone=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.attention1 = CBAM(256, reduction=16)
        
        self.layer2 = resnet.layer2
        self.attention2 = CBAM(512, reduction=16)
        
        self.layer3 = resnet.layer3
        self.attention3 = CBAM(1024, reduction=16)
        
        self.layer4 = self._make_layer4_dilated(resnet.layer4)
        self.attention4 = CBAM(2048, reduction=16)
        
        self.aspp = ASPP(2048, 256, dilations=[6, 12, 18])
        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)
    
    def _make_layer4_dilated(self, layer4):
        for n, m in layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        return layer4
    
    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.attention1(self.layer1(x))
        low_level_feat = x
        
        x = self.attention2(self.layer2(x))
        x = self.attention3(self.layer3(x))
        x = self.attention4(self.layer4(x))
        
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        valid_mask = (target != self.ignore_index)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target * valid_mask.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        valid_mask = valid_mask.unsqueeze(1).float()
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_one_hot = target_one_hot.reshape(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()

class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
    
    def _compute_distance_map(self, mask):
   
        h, w = mask.shape
        boundaries = np.zeros((h, w), dtype=np.uint8)    
        grad_x = np.abs(mask[:, :-1] - mask[:, 1:])
        grad_y = np.abs(mask[:-1, :] - mask[1:, :])

        boundaries[:, :-1] += (grad_x > 0).astype(np.uint8)
        boundaries[:-1, :] += (grad_y > 0).astype(np.uint8)
    
        boundaries = (boundaries > 0).astype(np.uint8)
    
        return boundaries
    
    def forward(self, pred, target):
        device = pred.device
        batch_size = pred.shape[0]
        pred = F.softmax(pred, dim=1)
        
        distance_maps = []
        for i in range(batch_size):
            target_np = target[i].cpu().numpy()
            valid_mask = target_np != self.ignore_index
            target_filtered = target_np.copy()
            target_filtered[~valid_mask] = 0
            dist_map = self._compute_distance_map(target_filtered)
            distance_maps.append(torch.from_numpy(dist_map).float())
        
        distance_maps = torch.stack(distance_maps).to(device)
        num_classes = pred.shape[1]
        valid_mask = (target != self.ignore_index)
        target_one_hot = F.one_hot(target * valid_mask.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        weight = 1.0 - distance_maps.unsqueeze(1)
        weight = weight * valid_mask.unsqueeze(1).float()
        loss = -target_one_hot * torch.log(pred + 1e-7) * weight
        
        return loss.sum() / (valid_mask.sum() + 1e-7)

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=104, ignore_index=255, 
                 ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        total_loss = (self.ce_weight * ce + 
                     self.dice_weight * dice + 
                     self.boundary_weight * boundary)
        return total_loss, {
            'ce_loss': ce.item(),
            'dice_loss': dice.item(),
            'boundary_loss': boundary.item(),
            'total_loss': total_loss.item()
        }

class SegmentationMetrics:
    def __init__(self, num_classes=104, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def _fast_hist(self, pred, target, num_classes):
        mask = (target >= 0) & (target < num_classes)
        hist = np.bincount(
            num_classes * target[mask].astype(int) + pred[mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return hist
    
    def update(self, pred, target):
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().reshape(-1)
        target = target.cpu().numpy().reshape(-1)
        
        valid_mask = (target != self.ignore_index)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        self.confusion_matrix += self._fast_hist(pred, target, self.num_classes)
        self.correct_pixels += np.sum(pred == target)
        self.total_pixels += len(target)
    
    def get_pixel_accuracy(self):
        if self.total_pixels == 0:
            return 0.0
        return self.correct_pixels / self.total_pixels
    
    def get_mean_iou(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-10)
        valid_classes = (tp + fn) > 0
        mean_iou = np.mean(iou[valid_classes])
        per_class_iou = {i: iou[i] for i in range(self.num_classes) if valid_classes[i]}
        return mean_iou, per_class_iou
    
    def get_metrics(self):
        pixel_acc = self.get_pixel_accuracy()
        mean_iou, per_class_iou = self.get_mean_iou()
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'per_class_iou': per_class_iou
        }
class FoodSeg103Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=320, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        self.img_dir = os.path.join(root_dir, 'Images', 'img_dir', split)
        self.mask_dir = os.path.join(root_dir, 'Images', 'ann_dir', split)
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, image, mask):
        if torch.rand(1) > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if torch.rand(1) > 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if torch.rand(1) > 0.5:
            angle = torch.rand(1).item() * 30 - 15
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        
        image = TF.resize(image, (self.img_size, self.img_size), 
                         interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.img_size, self.img_size), 
                        interpolation=transforms.InterpolationMode.NEAREST)
        
        if self.augment:
            image, mask = self._augment(image, mask)
        
        image = self.normalize(TF.to_tensor(image))
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

DATASET_ROOT = '/kaggle/input/foodseg103/FoodSeg103'

CONFIG = {
    'num_classes': 104,
    'img_size': 320,
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'patience': 10,
    'num_workers': 2,
    'ce_weight': 1.0,
    'dice_weight': 1.0,
    'boundary_weight': 0.0,
}

train_dataset = FoodSeg103Dataset(
    root_dir=DATASET_ROOT,
    split='train',
    img_size=CONFIG['img_size'],
    augment=True
)

val_dataset = FoodSeg103Dataset(
    root_dir=DATASET_ROOT,
    split='test',
    img_size=CONFIG['img_size'],
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
model = FoodSegmentationModel(
    num_classes=CONFIG['num_classes'],
    pretrained_backbone=True
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params / 1e6:.2f}M")

criterion = CombinedLoss(
    num_classes=CONFIG['num_classes'],
    ce_weight=CONFIG['ce_weight'],
    dice_weight=CONFIG['dice_weight'],
    boundary_weight=CONFIG['boundary_weight']
)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
from torch.cuda.amp import autocast, GradScaler

best_miou = 0.0
patience_counter = 0
history = {
    'train_loss': [], 'val_loss': [],
    'train_miou': [], 'val_miou': [],
    'train_pixel_acc': [], 'val_pixel_acc': []
}

os.makedirs('./checkpoints', exist_ok=True)
checkpoint_path = './checkpoints/best_model.pth'
start_epoch = 1

# AMP scaler
scaler = GradScaler()

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    print(f"--- Loading Checkpoint from {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint.get('history', history)
    best_miou = checkpoint.get('miou', 0.0)
    print(f"Successfully loaded! Resuming from Epoch {start_epoch}")
else:
    print("No checkpoint found. Starting from scratch.")

for epoch in range(start_epoch, CONFIG['num_epochs'] + 1):

    # ===================== TRAIN =====================
    model.train()
    train_metrics = SegmentationMetrics(num_classes=CONFIG['num_classes'])
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---- AMP Forward ----
        with autocast():
            outputs = model(images)
            loss, _ = criterion(outputs, masks)

        # ---- AMP Backward ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_metrics.update(outputs.detach(), masks.detach())
        train_loss += loss.item()

        metrics = train_metrics.get_metrics()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mIoU': f"{metrics['mean_iou']:.4f}"
        })

    train_loss /= len(train_loader)
    train_results = train_metrics.get_metrics()

    # ===================== VALIDATION =====================
    model.eval()
    val_metrics = SegmentationMetrics(num_classes=CONFIG['num_classes'])
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss, _ = criterion(outputs, masks)

            val_metrics.update(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_results = val_metrics.get_metrics()

    # ===================== HISTORY =====================
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_miou'].append(train_results['mean_iou'])
    history['val_miou'].append(val_results['mean_iou'])
    history['train_pixel_acc'].append(train_results['pixel_accuracy'])
    history['val_pixel_acc'].append(val_results['pixel_accuracy'])

    # LR Scheduler
    scheduler.step(val_results['mean_iou'])

    # ===================== LOG =====================
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Train mIoU: {train_results['mean_iou']:.4f} | Val mIoU: {val_results['mean_iou']:.4f}")
    print(f"  Train Pix Acc: {train_results['pixel_accuracy']:.4f} | Val Pix Acc: {val_results['pixel_accuracy']:.4f}")

    # ===================== CHECKPOINT =====================
    if val_results['mean_iou'] > best_miou:
        best_miou = val_results['mean_iou']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'miou': best_miou,
            'history': history
        }, checkpoint_path)

        print(f"  ✓ Best model saved! mIoU: {best_miou:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print("\n" + "=" * 80)
print(f"Training Complete! Best mIoU: {best_miou:.4f}")
print("=" * 80)

# ==================== CELL 1 ====================
def inference(image_path, model_path, img_size=320):
    model = FoodSegmentationModel(num_classes=104)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    original = image.copy()

    image = TF.resize(image, (img_size, img_size))
    image = TF.to_tensor(image)
    image = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='jet')
    plt.title("Segmentation Output")
    plt.axis('off')

    plt.show()

