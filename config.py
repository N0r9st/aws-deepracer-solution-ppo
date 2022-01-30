class config:
    ports = [8887, 8886, 8885, 8884, 8883, 8882]
    runs_in_test = 2
    num_stack = 4
    num_skip = 1
    image_shape = (64, 64)
    total_frames = 6_000_000
    actor_steps = 256
    batch_size = 256
    learning_rate = 2.5e-4

    backbone = 'SCLight'

    decaying_lr_and_clip_param = True
    gamma = .99
    lambda_ = .95
    clip_param = .1
    num_epochs = 3
    vf_coeff = 0.5
    entropy_coeff = 0.01

    checkpoint_frequency = 50
    test_frequency = 4
    
    run_name = f'{len(ports)}env_{backbone}_{num_stack}st{num_skip}sk'
    model_dir = './save/'

    load_state = None
    save = True
    wandb = True

    obs_shape = image_shape + (num_stack,)