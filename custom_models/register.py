from mmdet.registry import MODELS
from .custom_cascade_with_meta import CustomCascadeWithMeta
from .custom_heads import FCHead, RegHead

# Register the custom modules
MODELS.register_module(module=CustomCascadeWithMeta)
MODELS.register_module(module=FCHead)
MODELS.register_module(module=RegHead) 