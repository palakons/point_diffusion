import wandb

# Example sweep configuration
sweep_configuration = {
    "description": " na",
    "method": "grid",
    "name": "name",
    "parameters": {
        "data_dir": {"values": ["/data/palakons/dataset/astyx_blank/scene/"]},
        "batch_size": {"values": [2,8,32,128]},
        "lr": {"values": [1e-3,1e-4,1e-5]},
        "N": {"values": [1024]},
        "M": {"values": [1]},

    },
    "program": "from_scratch/dm.py",
    # "command": ["${env}","${interpreter}","${program}","${args_no_hyphens}"]
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="point_cloud_diffusion")
print("sweep_id:", sweep_id)