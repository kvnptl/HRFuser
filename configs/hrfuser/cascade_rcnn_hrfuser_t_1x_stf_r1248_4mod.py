_base_ = [
    '../_base_/models/cascade_rcnn_hrfuser_fpn_stf_clrg_fusion.py',
    '../_base_/datasets/kitti_detection_2d_c1248_clrg_fusion.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(
    backbone=dict(
        type='HRFuserHRFormerBased',
        drop_path_rate=0.,
        mod_in_channels=[3, 2, 1],
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
# # AdamW optimizer, no weight decay for position embedding & layer norm
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# runner=dict(max_epochs=60)
# lr_config=dict(policy='step', step=[40,50])
# data=dict(samples_per_gpu= 3, workers_per_gpu= 2)
# seed=0

'''
NOTE: 
- original setting is 4 GPUs, sample_per_gpu=3 => Total batch size=12 results in ==> lr=0.0001
- I am using 4 GPUs, sample_per_gpu=2 => Total batch size=8, so
  batch_size is reduced by (8/12), so have to adjust lr accordingly. This is according to Linear Scaling Rule paper.
- So, lr=0.0001 * (8/12) = 0.0006667 (new adjusted lr) 
'''
# -----> Settings 1, change lr, as per the Linear Scaling Rule and sample_per_gpu<-----
# AdamW optimizer, no weight decay for position embedding & layer norm
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006667,  # Adjusted learning rate
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
runner=dict(max_epochs=60)
lr_config=dict(policy='step', step=[40,50])
data=dict(samples_per_gpu=2, workers_per_gpu=2)  # Updated samples_per_gpu
seed=0

# -----> Settings 2 <-----
# Trying more workers per gpu, workers_per_gpu=3
# # AdamW optimizer, no weight decay for position embedding & layer norm
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# runner=dict(max_epochs=60)
# lr_config=dict(policy='step', step=[40,50])
# data=dict(samples_per_gpu= 3, workers_per_gpu= 3)
# seed=0