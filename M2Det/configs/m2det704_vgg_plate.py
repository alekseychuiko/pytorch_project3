model = dict(
    type = 'm2det',
    input_size = 704,
    init_net = True,
    pretrained = 'M2Det/weights/vgg16_reducedfc.pth',
    m2det_config = dict(
        backbone = 'vgg16',
        net_family = 'vgg', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        base_out = [22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = False,
        smooth = True,
        num_classes = 2,
        ),
    rgb_means = (104, 117, 123),
    p = 0.6,
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 117, 176], # if you want to set this with custom manner, compute by input size / each map's size.
        #size_pattern = [0.035, 0.085, 0.176, 0.352, 0.713, 0.99, 1.19],
        size_pattern = [0.04, 0.11, 0.23, 0.39, 0.58, 0.80, 1.05],
        ),
    save_eposhs = 5,
    weights_save = 'M2Det/weights/'
    )

train_cfg = dict(
    cuda = True,
    per_batch_size = 4,
    lr = [0.000005, 0.00001, 0.000005],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        ep = [20, 100, 150],
        ),
    print_epochs = 1,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
    keep_per_class = 50,
    save_folder = 'M2Det/eval'
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='Adam', momentum=0.9, weight_decay=0.0005)
