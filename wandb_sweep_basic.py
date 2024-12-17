import wandb

# Example sweep configuration
sweep_configuration = {
    "description": " na",
    "method": "grid",
    "name": "basic_dm 15 MLP LR no checkpoint 5M",
    "parameters": {
        #    Namespace(M=1, N=1000, batch_size=1, beta_end=0.008, beta_schedule='linear', beta_start=1e-05, checkpoint=None, checkpoint_freq=1000, config=None, data_dir='/data/palakons/dataset/astyx_blank/scene/', ema_decay=0.999, ema_update_freq=20, epochs=10000, hidden_dim=128, lr=0.001, model='PointCloudDiffusionModel', n_hidden_layers=1, no_wandb=False, num_time_steps=1000, num_time_steps_to_visualize=100, seed=42, start_epoch=0, visualize_freq=1000)
        "model": {"values": ["mlp3d"]},
        "beta_schedule": {
            "values": ["linear"]},  # ,"scaled_linear",  "squaredcos_cap_v2"]},
        "batch_size": {"values": [1]},
        "lr": {"values": [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]},
        "N": {"values": [100]},
        "M": {"values": [1]},
        "hidden_dim": {"values": [256]},
        "n_hidden_layers": {"values": [3]},
        "epochs": {"values": [5000000]},
        "num_train_timesteps": {"values": [1000]},
        "loss_type": {"values": ["mse"]},  # "emd","chamfer"]},
        "visualize_freq": {"values": [10000]},
        # "extra_channels": {"values": [1, 3, 5, 7]},
        "no_checkpoint": {"values": [True]},
    },
    "program": "from_scratch/basic_dm_15.py",
    # "command": ["${env}","${interpreter}","${program}","${args_no_hyphens}"]
}

sweep_id = wandb.sweep(sweep=sweep_configuration,
                       project="point_cloud_diffusion")
print("sweep_id:", sweep_id)
