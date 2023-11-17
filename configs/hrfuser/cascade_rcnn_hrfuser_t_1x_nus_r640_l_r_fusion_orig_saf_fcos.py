_base_ = [
    '../_base_/models/cascade_rcnn_hrfuser_fpn_nus_clr_fusion_saf_fcos.py',
    '../_base_/datasets/nuscenes_detection_r640_clr_fusion_saf_fcos.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(
    backbone=dict(
        type='HRFuserHRFormerBased',
        drop_path_rate=0.,
        extra=dict(
            ModFusionA=dict(
                num_channels=(18, 36)),
            LidarStageB=dict(
                num_channels=(18,)),
            ModFusionB=dict(
                num_channels=(18, 36, 72)),
            LidarStageC=dict(
                num_modules=3,
                num_channels=(18,)),
            ModFusionC=dict(
                num_channels=(18, 36, 72, 144)),
            # LidarStageD=dict(
            #     num_channels=(18,)),
            # ModFusionD=dict(
            #     num_channels=(18, 36, 72, 144)),
            stage2=dict(
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=3,
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_channels=(18, 36, 72, 144)))),
    neck=dict(
        in_channels=[18, 36, 72, 144]))

# ----> ORIG SETTING <-----
# AdamW optimizer, no weight decay for position embedding & layer norm
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
data=dict(samples_per_gpu= 3, workers_per_gpu= 2)
seed=8

# # ----> SETTING 1 <-----
# '''
# - Derived from ORIG SETTING
# - ONLY FOR 1 GPU
# - Learning rate changed according to Linear Scaling Rule

# - original setting is 4 GPUs, sample_per_gpu=3 => Total batch size=12 results in ==> lr=0.0003
# - I am using 1 GPUs, sample_per_gpu=3 => Total batch size=3, so
#   batch_size is reduced by (3/12), so have to adjust lr accordingly. This is according to Linear Scaling Rule paper.
# - So, lr=0.0003 * (3/12) = 0.000075 (new adjusted lr) 

# '''
# # AdamW optimizer, no weight decay for position embedding & layer norm
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.000075,  # Adjusted learning rate
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# data=dict(samples_per_gpu= 4, workers_per_gpu= 4)  # Changed
# seed=8

# ----> SETTING 2 <-----
# As mentioned in the paper, LR=0.0001
# AdamW optimizer, no weight decay for position embedding & layer norm
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# data=dict(samples_per_gpu= 3, workers_per_gpu= 2)
# seed=8