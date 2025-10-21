import cv2
import aruco
import pickle
import numpy as np

def draw_line(img, pt1, pt2, color, thickness=3):
    pt1 = (np.round(pt1[0]).astype(int), np.round(pt1[1]).astype(int))
    pt2 = (np.round(pt2[0]).astype(int), np.round(pt2[1]).astype(int))
    ret = cv2.line(img, pt1, pt2, color, thickness)
    return ret


def plot_axis(
        image,
        calibration,
        pose,
        axis_len=100,
        right_handed=False,
        thickness=3,
        distorts=True
):
    pose_ok, rv, tv = pose
    if not pose_ok:
        return

    K, dist_coeffs = calibration

    z_dir = -1 if right_handed else 1

    axis_points = np.array([
        [0, 0, 0],  # Origin
        [axis_len, 0, 0],  # X-axis
        [0, axis_len, 0],  # Y-axis
        [0, 0, z_dir * axis_len]  # Z-axis
    ], dtype=np.float32)

    # Project 3D points to the 2D image plane

    if distorts:
        use_dist_coeffs = dist_coeffs
    else:
        use_dist_coeffs = None
    axis_points_proj, _ = cv2.projectPoints(axis_points, rv, tv, K, use_dist_coeffs)

    # Draw the axes on the image
    axis_points_proj = axis_points_proj.reshape(-1, 2)
    origin = tuple(axis_points_proj[0].ravel())

    draw_line(image, origin, axis_points_proj[1], (0, 0, 255), thickness=thickness)
    draw_line(image, origin, axis_points_proj[2], (0, 255, 0), thickness=thickness)
    draw_line(image, origin, axis_points_proj[3], (255, 0, 0), thickness=thickness)

    return image

def read_calibration(calibration_file):
    with open(calibration_file, "rb") as f:
        raw = f.read()
        ret = pickle.loads(raw)
    return ret


if __name__ == "__main__":

    # read calibration
    stereo_calibration = read_calibration("res/stereo_calibration.pkl")
    left_calibration = stereo_calibration["left_K"], stereo_calibration["left_dist"]

    # read left capture
    image = cv2.imread('./res/left_0.jpg')

    cv2.imshow("original image", image)
    cv2.waitKey(0)

    # create charuco board
    board = aruco.create_charuco_board(
        squares_x=5,
        squares_y=7,
        square_length=52.6,
        marker_length=31.3,
    )

    # detect charuco board
    detection = aruco.detect_charuco_markers(image, board)

    # draw and show detection
    img_det = aruco.draw_charuco_markers(image, detection)
    cv2.imshow("detected charuco", img_det)
    cv2.waitKey(0)

    # estimate pose using my cool method
    pose = aruco.estimate_camera_pose_with_homography(
        image,
        board,
        detection,
        left_calibration,
    )

    image = plot_axis(image, left_calibration, pose)
    cv2.imshow("camera pose axis", image)
    cv2.waitKey(0)



    #image_size = img.shape[1], img.shape[0]
    #draw_charuco_markers
