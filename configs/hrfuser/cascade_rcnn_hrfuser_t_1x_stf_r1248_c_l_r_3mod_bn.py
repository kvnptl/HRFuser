_base_ = [
    './cascade_rcnn_hrfuser_t_1x_stf_r1248_c_l_r_3mod_orig.py'
]
norm_cfg = dict(type='BN', requires_grad=True, momentum=0.1)
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        norm_cfg=norm_cfg
    )
)
