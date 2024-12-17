from argparse import Namespace
import torch

latest_fname = "/data/palakons/checkpoint/cp_dm_2024-11-29-08-03-48.pth"

def process_checkpoint(checkpoint):
    print("Processing checkpoint")
    print(checkpoint)
    ckp = torch.load(checkpoint)
    
    # "model": model.state_dict(),
    # "optimizer": optimizer.state_dict(),
    # "args": args,
    return ckp

    print(ckp["args"])
def make_arguemnt_list(args: Namespace):
    print("Making argument list")
    print(args)
    return " --".join([f"{k}={v}" for k, v in vars(args).items()])

ckp=process_checkpoint(latest_fname)

print(make_arguemnt_list(ckp["args"]))