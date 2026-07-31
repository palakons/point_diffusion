from truckscenes import TruckScenes

tc_mini = TruckScenes("v1.0-mini", "/data/palakons/new_dataset/MAN/mini/man-truckscenes", False) 
tc_full = TruckScenes("v1.0-trainval", "/data/palakons/new_dataset/MAN/man-truckscenes", False) 

mini_scene_tokens = {s["token"] for s in tc_mini.scene}
full_scene_tokens = {s["token"] for s in tc_full.scene}

missing = mini_scene_tokens - full_scene_tokens

print(f"mini scenes: {len(mini_scene_tokens)}")
print(f"full scenes: {len(full_scene_tokens)}")
print(f"mini subset of full: {not missing}")

if missing:
    print("Missing mini scene tokens:")
    print(sorted(missing))


for key in ["token", "name"]:
    mini_vals = {s[key] for s in tc_mini.scene}
    full_vals = {s[key] for s in tc_full.scene}

    print(
        key,
        f"{len(mini_vals)}/{len(full_vals)}",
        "subset:", mini_vals <= full_vals,
        "missing:", mini_vals - full_vals,
    )