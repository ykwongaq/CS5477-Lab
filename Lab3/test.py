import argparse
import filecmp
import json
import os
import pickle

import cv2
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, choices=['temple', 'mini-temple'])
argparser.add_argument('--ba', action='store_true')
args = argparser.parse_args()

DATASET = args.dataset
SAVE_DIR = os.path.join(PREDICTION_DIR, DATASET)
KEYPOINT_DIR = os.path.join(SAVE_DIR, 'keypoints')
BF_MATCH_DIR = os.path.join(SAVE_DIR, 'bf-match')
BF_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'bf-match-images')

RANSAC_MATCH_DIR = os.path.join(SAVE_DIR, 'ransac-match')
RANSAC_ESSENTIAL_DIR = os.path.join(SAVE_DIR, 'ransac-fundamental')
RANSAC_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'ransac-match-images')
SCENE_GRAPH_FILE = os.path.join(SAVE_DIR, 'scene-graph.json')

SAVE_DIR = os.path.join(PREDICTION_DIR, DATASET)

HAS_BUNDLE_ADJUSTMENT = args.ba
SPLIT = 'bundle-adjustment' if HAS_BUNDLE_ADJUSTMENT else 'no-bundle-adjustment'
RESULT_DIR = os.path.join(SAVE_DIR, 'results', SPLIT)
ALL_EXTRINSIC = os.path.join(RESULT_DIR, 'all-extrinsic.json')
CORRESPONDENCES2D3D = os.path.join(RESULT_DIR, 'correspondences2d3d.json')
POINT3D_FILE = os.path.join(RESULT_DIR, 'points3d.npy')
REGISTRATION_TRAJECTORY = os.path.join(RESULT_DIR, 'registration-trajectory.txt')


def read_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def check_file_count(directory):
    ta = os.listdir(directory.replace("predictions", "ta-results"))
    prd = os.listdir(directory)
    return len(ta) == len(prd)


def check_keypoints():
    num_match = 0
    files = os.listdir(KEYPOINT_DIR)

    for file in files:
        file_path = os.path.join(KEYPOINT_DIR, file)
        ta = read_pickle(file_path.replace("predictions", "ta-results"))
        prd = read_pickle(file_path)
        keypoints_match = ta['keypoints'] == prd['keypoints']
        descriptors_match = np.all(ta['descriptors'] == prd['descriptors'])
        num_match += keypoints_match and descriptors_match

    content_match = num_match == len(files)
    file_count_match = check_file_count(KEYPOINT_DIR)
    match = content_match and file_count_match

    print("- {:30s}: {}".format('Keypoints', match))


def check_folder(directory, func):
    num_match = 0
    files = os.listdir(directory)
    folder_name = os.path.basename(directory)

    for file in files:
        file_path = os.path.join(directory, file)
        ta = func(file_path.replace("predictions", "ta-results"))
        prd = func(file_path)
        num_match += np.all(ta == prd)

    content_match = num_match == len(files)
    file_count_match = check_file_count(directory)
    match = content_match and file_count_match
    print("- {:30s}: {}".format(folder_name, match))


def check_npy_file(file_path, func):
    ta = func(file_path.replace("predictions", "ta-results"))
    prd = func(file_path)

    match = np.allclose(ta, prd, rtol=1e-04)
    file_name = os.path.basename(file_path)
    print("- {:30s}: {}".format(file_name, match))


def check_json_files(file_path):
    ta = file_path.replace("predictions", "ta-results")
    prd = file_path

    match = filecmp.cmp(ta, prd)
    file_name = os.path.basename(file_path)
    print("- {:30s}: {}".format(file_name, match))


def check_all_extrinsic_file(file_path):
    ta = read_json(file_path.replace("predictions", "ta-results"))
    prd = read_json(file_path)
    num_match = 0

    for key, val in ta.items():
        num_match += np.allclose(val, prd[key], rtol=1e-04)

    match = num_match == len(ta)
    file_name = os.path.basename(file_path)
    print("- {:30s}: {}".format(file_name, match))


def main():
    print("\npreprocess.py")
    check_folder(BF_MATCH_DIR, func=np.load)
    check_folder(BF_MATCH_IMAGE_DIR, func=cv2.imread)
    check_keypoints()
    check_folder(RANSAC_ESSENTIAL_DIR, func=np.load)
    check_folder(RANSAC_MATCH_DIR, func=np.load)
    check_folder(RANSAC_MATCH_IMAGE_DIR, func=cv2.imread)
    check_json_files(SCENE_GRAPH_FILE)

    print("\nsfm.py")
    check_all_extrinsic_file(ALL_EXTRINSIC)
    check_json_files(CORRESPONDENCES2D3D)
    check_npy_file(POINT3D_FILE, func=np.load)
    check_json_files(REGISTRATION_TRAJECTORY)


if __name__ == "__main__":
    main()
