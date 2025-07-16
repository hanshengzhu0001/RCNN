# -*- coding: utf-8 -*-
# File: cascade_rcnn_r50_fpn.py

# 1. Import settings: model code + pipelines
custom_imports = dict(
    imports=[
        'mmdet.models',
        'mmdet.datasets',
        'mmdet.datasets.pipelines',
        'mmdet.datasets.pipelines.transforms',
        'mmdet.datasets.pipelines.formatting',
        'mmdet.datasets.pipelines.loading',
        'mmdet.datasets.pipelines.formatting.formatting',
        'mmdet.datasets.pipelines.formatting.default_format_bundle',
        'mmdet.datasets.pipelines.formatting.packaging',
    ],
    allow_failed_imports=True
)
default_scope = 'mmdet'

# -------------------------------------------------------------------
# 2. Model settings
# -------------------------------------------------------------------
model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0] * 4,
            target_stds=[1.0] * 4
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        test_cfg=dict(
            nms_pre=1000,
            max_num=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        )
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,  # Total number of classes
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0] * 4,
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0] * 4,
                    target_stds=[0.05, 0.05, 0.1, 0.1]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0] * 4,
                    target_stds=[0.033, 0.033, 0.067, 0.067]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
        ],
        mask_roi_extractor=None,
        mask_head=None
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1
                ),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1
                ),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1
                ),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            ),
        ]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_num=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
    )
)

# -------------------------------------------------------------------
# 3. Dataset & pipelines
# -------------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction',
            'chart_type', 'plot_bb', 'axes_info', 'data_series'
        )
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction',
            'chart_type', 'plot_bb', 'axes_info', 'data_series'
        )
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=True, seed=42),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_JSON/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
        pipeline=train_pipeline,
        test_mode=False,
        metainfo=dict(
            classes=(
                'plot_area', 'line_segment', 'axis_tick', 'axis_label', 'chart_title',
                'scatter_point', 'bar_segment', 'dot_point', 'tick_label', 'axis_title',
                'chart_title', 'legend_title', 'legend_label'
            )
        )
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=True, seed=42),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_JSON/val_with_info.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(
            classes=(
                'plot_area', 'line_segment', 'axis_tick', 'axis_label', 'chart_title',
                'scatter_point', 'bar_segment', 'dot_point', 'tick_label', 'axis_title',
                'chart_title', 'legend_title', 'legend_label'
            )
        )
    )
)

# -------------------------------------------------------------------
# 4. Evaluator
# -------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_JSON/val_with_info.json',
    metric=['bbox'],
    classwise=True,
    proposal_nums=(100, 1000, 3000),
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)

# -------------------------------------------------------------------
# 5. Optimizer & schedulers
# -------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, begin=0, end=100, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=2, by_epoch=True, milestones=[1], gamma=0.1)
]

# -------------------------------------------------------------------
# 6. Training/Validation loops
# -------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
val_cfg   = dict(type='ValLoop')

# -------------------------------------------------------------------
# 7. Logging & checkpoint
# -------------------------------------------------------------------
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                project='chart-detection',
                name='cascade_rcnn_r50_fpn',
                config=dict(
                    model=model,
                    dataset=dict(
                        train=train_dataloader['dataset'],
                        val=val_dataloader['dataset']
                    )
                )
            ),
            interval=1,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3
        )
    ]
)

# -------------------------------------------------------------------
# 8. Runtime & work_dir
# -------------------------------------------------------------------
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/work_dir'

# -------------------------------------------------------------------
# 9. Default runtime settings
# -------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# -------------------------------------------------------------------
# 10. No custom hooks
# -------------------------------------------------------------------
custom_hooks = []
