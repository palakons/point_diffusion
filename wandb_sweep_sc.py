import wandb

# Example sweep configuration
sweep_configuration = {
    "description": " na",
    "method": "grid",
    "name": "hidden new_dm ",
    "parameters": {
        
        #    Namespace(M=1, N=1000, batch_size=1, beta_end=0.008, beta_schedule='linear', beta_start=1e-05, checkpoint=None, checkpoint_freq=1000, config=None, data_dir='/data/palakons/dataset/astyx_blank/scene/', ema_decay=0.999, ema_update_freq=20, epochs=10000, hidden_dim=128, lr=0.001, model='PointCloudDiffusionModel', n_hidden_layers=1, no_wandb=False, num_time_steps=1000, num_time_steps_to_visualize=100, seed=42, start_epoch=0, visualize_freq=1000)
        "data_dir": {"values": ["/data/palakons/dataset/astyx_blank/scene/"]},
        "model": {"values": ["PointCloudDiffusionModel"]},
        "batch_size": {"values": [1]},
        "lr": {"values": [1e-4]},
        "N": {"values": [1024]},
        "M": {"values": [1]},
        "epochs": {"values": [50000]},

        # add n_hidden_layers and hidden_dim
        "n_hidden_layers": {"values": [1,2,4]},
        "hidden_dim": {"values": [64,128,256]},
        "num_time_steps": {"values": [1000]},
        "num_time_steps_to_visualize": {"values": [100]},
        "visualize_freq": {"values": [1000]},
        "checkpoint_freq": {"values": [1000]},
                                 

    },
    "program": "from_scratch/dm_new.py",
    # "command": ["${env}","${interpreter}","${program}","${args_no_hyphens}"]
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="point_cloud_diffusion")
print("sweep_id:", sweep_id)