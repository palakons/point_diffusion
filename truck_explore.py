import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, trange
import numpy as np
import open3d as o3d
import time

from truckscenes import TruckScenes

# Directory to save images
output_dir = '/ist-nas/users/palakonk/singularity_logs/MAN_rendered_images'
os.makedirs(output_dir, exist_ok=True)

# Get the truck scene

trucksc = TruckScenes(
    'v1.0-mini', '/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/mini/man-truckscenes', True)
# trucksc_full = TruckScenes(
#     'v1.0-trainval', '/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/man-truckscenes', True)


def make_video(trucksc):
    scenes = trucksc.scene

    # Iterate over sequences
    for scene in tqdm(scenes):
        # Get the first frame of the sequence
        first_frame_token = scene['first_sample_token']

        # Render the point cloud in the image
        trucksc.render_pointcloud_in_image(
            first_frame_token, pointsensor_channel='RADAR_LEFT_FRONT',  dot_size=10)

        # Save the rendered image
        image_path = os.path.join(output_dir, f'{first_frame_token}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Create a video from the saved images using ffmpeg
    video_path = os.path.join(output_dir, 'output_video.mp4')
    # os.system(
    #     f"ffmpeg -framerate 1 -i {output_dir}/%*.png -c:v libx264 -pix_fmt yuv420p {video_path}")
    os.system(
        f"ffmpeg -framerate 1 -i {output_dir}/%*.png {video_path}")


def how_many_scenes(trucksc):
    return len(trucksc.scene)


def explore_scene(trucksc, scene):
    # print(scene)
    next_sample_token = scene['first_sample_token']
    c = 0
    first_timestamp = trucksc.get('sample', next_sample_token)['timestamp']
    while next_sample_token:
        my_sample = trucksc.get('sample', next_sample_token)
        next_sample_token = my_sample['next']
        print(c, next_sample_token,
              (my_sample['timestamp']-first_timestamp)/1000000)
        # dict_keys(['token', 'scene_token', 'timestamp', 'prev', 'next', 'data', 'anns'])
        # print(my_sample.keys())
        # print(my_sample['data'].keys())  # dict_keys(['RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE', 'RADAR_RIGHT_FRONT', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_LEFT_BACK', 'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR', 'CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK'])
        # 1e6375db490e4563b55fce389b06a53b
        # print(my_sample['data']['RADAR_LEFT_FRONT'])
        my_radar = trucksc.get(
            'sample_data', my_sample['data']['RADAR_LEFT_FRONT'])
        # dict_keys(['token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next', 'sensor_modality', 'channel'])
        # print(my_radar.keys())#dict_keys(['token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next', 'sensor_modality', 'channel'])
        print(my_radar)  # {'token': '1e6375db490e4563b55fce389b06a53b', 'sample_token': '32d2bcf46e734dffb14fe2e0a823d059', 'ego_pose_token': '9f5d4bc97327401cabe5726c0deb153e', 'calibrated_sensor_token': '5e6c5afaa842478db6066e9de8dec1ef', 'timestamp': 1695473372704727, 'fileformat': 'pcd', 'is_key_frame': True, 'height': 1, 'width': 800, 'filename': 'samples/RADAR_LEFT_FRONT/RADAR_LEFT_FRONT_1695473372704727.pcd', 'prev': 'c03850fc93fe413c92529dd6086cf91a', 'next': '1ff031d024cd4c86b51ea2e1568761b0', 'sensor_modality': 'radar', 'channel': 'RADAR_LEFT_FRONT'}
        c_radar = 0
        current_radar_token = my_radar['next']
        while current_radar_token:
            my_radar_2 = trucksc.get('sample_data', current_radar_token)
            current_radar_token = my_radar_2['next']
            print(c_radar, current_radar_token,
                  my_radar_2['timestamp'], my_radar_2['is_key_frame'])
            c_radar = c_radar+1

            # print(my_radar)
            # break
        current_radar_token = my_radar['prev']
        while current_radar_token:
            my_radar_2 = trucksc.get('sample_data', current_radar_token)
            current_radar_token = my_radar_2['prev']
            print(c_radar, current_radar_token,
                  my_radar_2['timestamp'], my_radar_2['is_key_frame'])
            c_radar = c_radar+1

        return
        c = c+1
    # print(my_sample.keys())
    # dict_keys(['token', 'scene_token', 'timestamp', 'prev', 'next', 'data', 'anns'])


def available_sensors(trucksc):
    return [a['channel'] for a in trucksc.sensor]


def make_video_one_sc(trucksc, first_frame_token, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    frame_token = first_frame_token
    while frame_token != "":

        # Render the point cloud in the image
        trucksc.render_pointcloud_in_image(
            frame_token, pointsensor_channel='RADAR_LEFT_FRONT',  dot_size=10, camera_channel='CAMERA_RIGHT_FRONT')

        # Save the rendered image
        image_path = os.path.join(output_dir, f'{i:02d}-{frame_token}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        frame_token = trucksc.get('sample', frame_token)['next']
        # print(frame_token)
        i = i + 1

    # Create a video from the saved images using ffmpeg
    video_path = os.path.join(
        output_dir, f'output_video-{first_frame_token}.mp4')
    # make video quality super high
    os.system(
        f"ffmpeg -framerate 1 -i {output_dir}/%*.png {video_path} -c:v libx264 -pix_fmt yuv720p")
    # rm any png files in the output_dir
    os.system(f"rm {output_dir}/*.png")


def frame_count_seq_lnegth(trucksc, sc_id, sensor='RADAR_LEFT_FRONT'):

    scene = trucksc.scene[sc_id]
    next_sample_token = scene['first_sample_token']
    my_sample = trucksc.get('sample', next_sample_token)
    next_radar_token = my_sample['data'][sensor]
    c = 0
    c_key_frame = 0
    while trucksc.get('sample_data', next_radar_token)["prev"]:
        # print(".", end='')
        next_radar_token = trucksc.get('sample_data', next_radar_token)["prev"]
        # print("next_radar_token", next_radar_token)
    st_timestamp = trucksc.get(
        'sample_data', next_radar_token)['timestamp']
    # print("sample_data", trucksc.get(
    # 'sample_data', next_radar_token))
    while next_radar_token:
        try:
            my_radar = trucksc.get('sample_data', next_radar_token)
        except Exception as e:
            print("error", e)
            end_timestamp = my_radar['timestamp']
            break

        # print(c, next_radar_token, my_radar['is_key_frame'])
        next_radar_token = my_radar['next']
        c = c+1
        if my_radar['is_key_frame']:
            c_key_frame = c_key_frame + 1
        # print(my_radar)
        end_timestamp = my_radar['timestamp']
    return c, c_key_frame, (end_timestamp-st_timestamp)/1000000


# make_video(trucksc)
# explore_scene(trucksc, trucksc.scene[0])

n = how_many_scenes(trucksc)
print("how many scenes", n)

# print("available sensors", available_sensors(trucksc))  # 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE', 'RADAR_RIGHT_FRONT', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_LEFT_BACK', 'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR', 'CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK', 'XSENSE_CABIN', 'XSENSE_CHASSIS'
big_count_list = []
for sensor in ['RADAR_LEFT_FRONT', 'RADAR_LEFT_BACK', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE']:
    count_list = []
    for i in range(n):
        count_list.append(frame_count_seq_lnegth(
            trucksc, i, sensor=sensor))
    sum_count = np.sum(count_list, axis=0)
    big_count_list.append(sum_count)
    print(sensor, "frame count", sum_count)
    # print(sensor, "frame count", frame_count_seq_lnegth(
    # trucksc, 0, sensor=sensor))
big_sum = np.sum(big_count_list, axis=0)
# print("big_sum_racar", big_sum)
print("radar frame count", big_sum[0], " key frame", big_sum[1], "duration (h:mm:ss)",
      big_sum[2]//3600, ":", (big_sum[2]//60) % 60, ":", big_sum[2] % 60)

big_count_list = []
for sensor in ['CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK']:
    count_list = []
    for i in range(n):
        count_list.append(frame_count_seq_lnegth(
            trucksc, i, sensor=sensor))
    sum_count = np.sum(count_list, axis=0)
    # print(sensor, "frame count", sum_count)
    big_count_list.append(sum_count)
big_sum = np.sum(big_count_list, axis=0)
# print("big_sum_cam", big_sum)
print("camera frame count", big_sum[0], " key frame", big_sum[1], "duration (h:mm:ss)",
      big_sum[2]//3600, ":", (big_sum[2]//60) % 60, ":", big_sum[2] % 60)
if False:
    summarize_list = []
    for i in trange(n):
        for pointsensor_channel in ['RADAR_LEFT_FRONT', 'RADAR_LEFT_BACK', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE']:
            for camera_channel in ['CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK']:
                my_sample = trucksc.get(
                    'sample', trucksc.scene[i]['first_sample_token'])
                points = trucksc.render_pointcloud_in_image(
                    my_sample['token'], pointsensor_channel=pointsensor_channel, camera_channel=camera_channel, dot_size=2)
                # print("points", points.shape[1], pointsensor_channel, camera_channel)
                plt.close()
                summarize_list.append({
                    'pointsensor_channel': pointsensor_channel,
                    'camera_channel': camera_channel,
                    'points_count': points.shape[1]
                })
    # print pivot/sum of the points count, using maybe pandas
    df = pd.DataFrame(summarize_list)
    # sum and ,mean of points count, for each combination of pointsensor_channel and camera_channel
    df_grouped = df.groupby(
        ['pointsensor_channel', 'camera_channel']).agg({'points_count': ['sum', 'mean']}).reset_index()
    df_grouped.columns = ['pointsensor_channel', 'camera_channel',
                          'points_count_sum', 'points_count_mean']
    print(df_grouped)


scene = trucksc.scene[0]
first_frame_token = scene['first_sample_token']
frame_token = first_frame_token
i = 0
while False and frame_token != "":

    for fname, radar_sensor, camera_sensor in zip(["green", "yellow", "red"], ['RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_LEFT_BACK'],
                                                  ['CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK']):

        # Render the point cloud in the image
        trucksc.render_pointcloud_in_image(
            frame_token, pointsensor_channel=radar_sensor, camera_channel=camera_sensor, dot_size=10)

        # Save the rendered image
        image_path = os.path.join(output_dir, f'{i}-{frame_token}-{fname}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    frame_token = trucksc.get('sample', frame_token)['next']
    # print(frame_token)
    i = i + 1
if False:
    output_dir_sc_video = os.path.join(
        output_dir, f'output_video-{first_frame_token}')
    make_video_one_sc(trucksc, first_frame_token, output_dir_sc_video)


# return sensor,order,token,x,y,z, is_key_frame
def list_points(trucksc, scene, root, sensors, header_attr):

    data = []
    next_sample_token = scene['first_sample_token']
    # next_sample_token = scene['last_sample_token']
    try:
        print("checking token", next_sample_token)
        my_sample = trucksc.get('sample', next_sample_token)
    except Exception as e:
        print("error sample", scene)
        return []
    for sensor in tqdm(sensors, desc="Sensors", leave=False):
        next_radar_token = my_sample['data'][sensor]
        while trucksc.get('sample_data', next_radar_token)["prev"]:
            print("<", end='')
            next_radar_token = trucksc.get(
                'sample_data', next_radar_token)["prev"]
        order = 0
        while next_radar_token:
            my_radar = trucksc.get('sample_data', next_radar_token)
            # try:
            #     my_radar = trucksc.get('sample_data', next_radar_token)
            # except Exception as e:
            #     print("error", e)
            #     break
            data_line = {**header_attr, **{"scene": scene['first_sample_token'], 'sensor': sensor, 'order': order,
                         'token': my_radar['token'],  'is_key_frame': my_radar['is_key_frame']}}
            # read pcd file
            pcd_file = os.path.join(root, my_radar['filename'])
            pcd = o3d.io.read_point_cloud(pcd_file)
            for i in range(len(pcd.points)):
                data_line = {
                    **{'x': pcd.points[i][0], 'y': pcd.points[i][1], 'z': pcd.points[i][2]}, **data_line}
                data.append(data_line)
            order = order+1
            next_radar_token = my_radar['next']

    return data


def points_to_csv(datasets, csv_fname):  # for mean std
    data_list = []
    sensors = ['RADAR_LEFT_FRONT', 'RADAR_LEFT_BACK', 'RADAR_LEFT_SIDE',
               'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE']
    for d_key, root in datasets.items():
        print("Dataset:", d_key)
        trucksc = TruckScenes(
            d_key, root, True)
        tt = tqdm(trucksc.scene, desc="Scenes", leave=False)
        for scene in tt:
            print("Scene:", scene['token'])
            # Get the first frame of the sequence
            sc_data = list_points(trucksc,
                                  scene, root, sensors, {"dataset": d_key})

            data_list += sc_data

            time_st = time.time()
            df = pd.DataFrame(data_list)
            # save to csv
            time_st2 = time.time()
            df.to_csv(csv_fname, index=False)
            # save to csv
            # tt.set_description(
            # f"{len(sc_data)} points , time to save {time.time()-time_st2:.2f} s, time to convert {time_st2-time_st:.2f} s")


points_to_csv({
    # "v1.0-mini":  "/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/mini/man-truckscenes",
    "v1.0-trainval": "/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/man-truckscenes"
},
    "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_full.csv")


# combine csv "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_full.csv" and "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_mini.csv" into one csv file
df_full = pd.read_csv(
    "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_full.csv")
df_mini = pd.read_csv(
    "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_mini.csv")
df = pd.concat([df_full, df_mini], ignore_index=True)


def print_stat(df):
    print("Summary of the data(all):")
    # mean and std of x,y,z
    print("Mean:", df['x'].mean(), df['y'].mean(), df['z'].mean())
    print("Std:", df['x'].std(), df['y'].std(), df['z'].std())

    # of only key frames
    df_kf = df[df['is_key_frame'] == True]
    print("Summary of the data(keyframe):")
    # mean and std of x,y,z
    print("Mean:", df_kf['x'].mean(), df_kf['y'].mean(), df_kf['z'].mean())
    print("Std:", df_kf['x'].std(), df_kf['y'].std(), df_kf['z'].std())

    # of only non key frames
    df_nkf = df[df['is_key_frame'] == False]
    print("Summary of the data(non-keyframe):")
    # mean and std of x,y,z
    print("Mean:", df_nkf['x'].mean(), df_nkf['y'].mean(), df_nkf['z'].mean())
    print("Std:", df_nkf['x'].std(), df_nkf['y'].std(), df_nkf['z'].std())


print("Stat for MINI:")
print_stat(df_mini)
print("Stat for FULL:")
print_stat(df_full)
print("Stat for ALL:")
print_stat(df)
