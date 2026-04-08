
set -euo pipefail
session="sweep"
image="/ist-nas/users/palakonk/singularity/sand"
binds="--bind /ist-nas/users/palakonk/singularity/home/$USER:/home/$USER --bind /ist-nas/users/palakonk/singularity_data:/data --bind /ist-nas/users/palakonk/singularity_logs:/home/$USER/logs --bind /tmp:/tmp --bind /ist-nas/ist-share/vision:/checkpoints_nas --bind /ist/ist-share/vision:/checkpoints --bind /ist/ist-share/vision/comfyui/models/checkpoints:/checkpoints_comfyui"

IFS=',' read -r -a devices <<< "0,1,2,3"
read -r -a factors <<< "43 44 45 46"

# Create session and main window if not exists
if ! tmux has-session -t "$session" 2>/dev/null; then
  tmux new-session -d -s "$session" -n "sweep"
fi

main_window="s_$RANDOM"

for i in "${!factors[@]}"; do  
  v="${factors[$i]}"
  dev="${devices[$(( i % ${#devices[@]} ))]}"

  echo "Starting run with factor $v on device $dev in tmux pane $i"

  # base inner command
  inner_cmd="source ~/.bashrc && conda activate pro_pt3d && CUDA_VISIBLE_DEVICES=${dev} python /home/palakons/point_diffusion/ptv3_adaln0_ddpm.py --num_input_frames=2 --lr_scheduler constant --dit_epochs 10000 --exp_name normal_800pt-dmode3-sd${v} --dit_lr  1e-4 --num_points 800 --seed_model ${v} --batch_size 1 --plot_every 500 --gpu_log_every 50 --ptv3_grid_size .1  --num_inference_steps 50 --ptv3_n_stages 2 --debug_mode 3"

  # full command runs inside singularity
  full_cmd="singularity exec --containall --nv $binds $image bash -lc \"${inner_cmd}\""

  if [ "$i" -eq 0 ]; then
    # First run: use the main window's first pane
    tmux send-keys -t "$session:$main_window.0" "$full_cmd" C-m
  else
    # Split new pane and run
    tmux split-window -t "$session:$main_window" -h
    tmux send-keys -t "$session:$main_window.$i" "$full_cmd" C-m
    tmux select-layout -t "$session:$main_window" tiled
  fi
  sleep 0.2
done

tmux select-window -t "$session:$main_window"
tmux attach -t "$session"