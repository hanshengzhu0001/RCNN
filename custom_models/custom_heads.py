from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module(name='FCHead')
class FCHead(nn.Module):
    """Simple fully connected head for classification."""
    
    def __init__(self, in_channels, num_classes, loss=None, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.loss_cfg = loss  # Optionally store, but not used
        
    def forward(self, x, target=None):
        # Global average pooling
        if isinstance(x, (list, tuple)):
            x = x[-1]  # Use the last feature map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.fc(x)
        
        if target is not None:
            loss = F.cross_entropy(logits, target)
            return loss
        return logits

@MODELS.register_module(name='RegHead')
class RegHead(nn.Module):
    """Simple fully connected head for regression."""
    
    def __init__(self, in_channels, out_dims, loss=None, max_points=None, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_dims)
        self.loss_cfg = loss  # Optionally store, but not used
        self.max_points = max_points  # Optionally store, but not used
        
    def forward(self, x, target=None):
        # Global average pooling
        if isinstance(x, (list, tuple)):
            x = x[-1]  # Use the last feature map
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Regression
        pred = self.fc(x)
        
        if target is not None:
            loss = F.smooth_l1_loss(pred, target)
            return loss
        return pred 