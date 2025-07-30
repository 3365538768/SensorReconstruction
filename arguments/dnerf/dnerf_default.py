

OptimizationParams = dict(

    coarse_iterations=3000,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations=20000,
    pruning_interval=8000,
    percent_dense=0.01,
    render_process=False,
    # 增加模型保存频率 - 避免训练中断导致模型丢失
    save_iterations=[20000],
    # 增加测试评估频率 - 更好地监控训练进度
    test_iterations=[1000, 3000, 5000, 7000, 10000, 15000, 20000],
    # no_do=False,
    # no_dshs=False
    # opacity_reset_interval=30000

)

ModelHiddenParams = dict(

    multires=[1, 2, 4, 8],
    defor_depth=6,
    net_width=256,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    weight_decay_iteration=0,
    bounds=1.6
)
