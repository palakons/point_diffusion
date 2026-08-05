from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil,os

def main():

    args = os.sys.argv[1:]
    if args:
        BASE_DIR = Path(
            "/data/palakons/ddpm_cond_slow/"
            f"{args[0]}/"
            "inference"
        )
    
    cond_use = ['nn_retrieval','zero_cond','correct_cond','shuffled_cond',]
    #join 4 csvs
    df_all = pd.DataFrame()
    for c_name in cond_use:

        csv_file_path = os.path.join(BASE_DIR, f"sampled_stat_{c_name}_sd42.csv")
        assert os.path.exists(csv_file_path), f"CSV file not found: {csv_file_path}"
        per_frame_cds_df = pd.read_csv(csv_file_path)
        df_all = pd.concat([df_all, per_frame_cds_df])

    df_all = df_all[['xyz_cd','doppler_mae','rcs_mae','doppler_sign_accuracy','condition_use']]


    #accgregate, mean, based on condition_use
    df_all = df_all.groupby('condition_use').mean().reset_index()
    print(df_all)


    # data_file,sensor_side,scene_id,frame_index,token,condition_use,xyz_cd,centroid_error,range_hist_error,azm_hist_error,x_y_occupancy_error,binary_x_y_occupancy_error,doppler_mae,rcs_mae,doppler_hist_mae,rcs_hist_mae,doppler_sign_accuracy
    # man-mini,left,5,4,6e169e7d226d47ba88c32424fc414937,shuffled_cond,109.47579193115234,8.681060791015625,4.5,5.3,0.055,0.035,21.095048904418945,13.843953132629395,6.4,6.85,1.0
        


if __name__ == "__main__":
    main()