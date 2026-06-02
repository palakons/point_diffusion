import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, tb_dir, data_dir,config):
        self.tb_dir = Path(tb_dir)
        self.data_dir = Path(data_dir)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tb = SummaryWriter(self.tb_dir )
        self.train_jsonl = open(self.data_dir / "train_log.jsonl", "a")
        self.val_jsonl = open(self.data_dir / "val_log.jsonl", "a")

        self.save_config(config)

    def save_config(self, config):
        config_path = self.data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        self.tb.add_text("config", json.dumps(config, indent=2), global_step=0)

    def log_scalars(self, prefix, step, data):
        clean = {}
        for k, v in data.items():
            if hasattr(v, "detach"):
                v = v.detach().float().mean().item()
            elif isinstance(v, int):
                v = float(v)
            elif isinstance(v, float):
                pass
            else:
                continue

            tag = f"{prefix}/{k}"
            self.tb.add_scalar(tag, v, step)
            clean[k] = v

        return clean
    def log_grouped_scalars(self, group_name, step, data):
        clean = {}
        for k, v in data.items():
            if hasattr(v, "detach"):
                v = v.detach().float().mean().item()
            elif isinstance(v, int):
                v = float(v)
            elif not isinstance(v, float):
                continue
            clean[k] = v
        self.tb.add_scalars(group_name, clean, step)
        return clean
    def log_train(self, step, data,log_group=True):
        #use log_grouped_scalars sepratedly for keys ending with std,mean,loss,mae,hist_error,occupancy_error,cd
        if log_group:
            for suffix in ["std", "mean", "loss", "mae", "hist_error", "occupancy_error", "cd"]:
                group_data = {k: v for k, v in data.items() if k.endswith(suffix)}
                if group_data:
                    self.log_grouped_scalars(f"train/{suffix}", step, group_data)
        clean = self.log_scalars("train", step, data)
        clean["step"] = step
        self.train_jsonl.write(json.dumps(clean) + "\n")
        self.train_jsonl.flush()

    def log_val(self, step, data , log_group=True):
        #use log_grouped_scalars sepratedly for keys ending with std,mean,loss,mae,hist_error,occupancy_error,cd
        if log_group:
            for suffix in ["std", "mean", "loss", "mae", "hist_error", "occupancy_error", "cd"]:
                group_data = {k: v for k, v in data.items() if k.endswith(suffix)}
                if group_data:
                    self.log_grouped_scalars(f"val/{suffix}", step, group_data)
        clean = self.log_scalars("val", step, data)
        clean["step"] = step
        self.val_jsonl.write(json.dumps(clean) + "\n")
        self.val_jsonl.flush()

    def close(self):
        self.tb.close()
        self.train_jsonl.close()
        self.val_jsonl.close()