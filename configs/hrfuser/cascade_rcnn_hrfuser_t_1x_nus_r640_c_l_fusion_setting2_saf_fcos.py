_base_ = [
    '../_base_/models/cascade_rcnn_hrfuser_fpn_nus_cl_fusion_saf_fcos.py',
    '../_base_/datasets/nuscenes_detection_r640_cl_fusion_saf_fcos.py', '../_base_/default_runtime.py',
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
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0003,
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

# # ----> SETTING 1 <-----
# # increased max_epochs from 12 to 36, and lr step changes from 8,11 to 27,33
# # AdamW optimizer, no weight decay for position embedding & layer norm
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0003,  # Adjusted learning rate
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# data=dict(samples_per_gpu= 3, workers_per_gpu= 8)  # Changed
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[27, 33])
# runner = dict(type='EpochBasedRunner', max_epochs=36)
# seed=8

# ----> SETTING 2 <-----
# increased max_epochs from 12 to 36, and lr step changes from 8,11 to 27,33
# batch size, samples per gpu 12, so total 48, adjusted lr=0.012
'''
NOTE: 
- original setting is 4 GPUs, sample_per_gpu=3 => Total batch size=12 results in ==> lr=0.0003
- I am using 4 A100 GPUs, sample_per_gpu=12 => Total batch size=48, so
  batch_size is increased by (48/12), so have to adjust lr accordingly. This is according to Linear Scaling Rule paper.
- So, lr=0.0003 * (48/12) = 0.012 (new adjusted lr) 
'''
# AdamW optimizer, no weight decay for position embedding & layer norm
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0012,  # Adjusted learning rate
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
data=dict(samples_per_gpu= 12, workers_per_gpu= 8)  # Changed
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
seed=8