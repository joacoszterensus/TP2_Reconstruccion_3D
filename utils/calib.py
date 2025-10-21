import numpy as np
import cv2
import glob


def draw_checkerboard(
        image,
        board_size,
        corners,
        found,
        line_thickness=2,
        corner_radius=5,
        corner_thickness=2,
        line_color=(255, 0, 255),
        circles_color=(0, 255, 255)
):
    """
        Draws detected checkerboard.

        Parameters:
        - image: The image where the corners will be drawn.
        - corners: The detected corners from cv2.findChessboardCorners.
        - board_size: The size of the chessboard (rows, columns).
        - line_thickness: Thickness of the lines connecting the corners.
        - corner_radius: Radius of the circles at each corner.
        - corner_thickness: Thickness of the circles at each corner.
        - color: Color of the lines and circles (B, G, R).
    """

    if not found:
        return image

    # Ensure corners are in integer format for drawing
    corners = corners.astype(int)

    # Draw lines connecting corners
    # line_color = color  # (0, 255, 0)
    # circles_color = (255, 0, 0)
    for i in range(board_size[1]):
        for j in range(board_size[0] - 1):
            idx1 = i * board_size[0] + j
            idx2 = i * board_size[0] + (j + 1)
            cv2.line(image, tuple(corners[idx1][0]), tuple(corners[idx2][0]), line_color, line_thickness)

    for i in range(board_size[1] - 1):
        for j in range(board_size[0]):
            idx1 = i * board_size[0] + j
            idx2 = (i + 1) * board_size[0] + j
            cv2.line(image, tuple(corners[idx1][0]), tuple(corners[idx2][0]), line_color, line_thickness)

    # Draw circles at each corner
    for corner in corners:
        cv2.circle(image, tuple(corner[0]), corner_radius, circles_color, corner_thickness)

    return image


def detect_board(CHECKERBOARD, gray, criteria=None, subpix_win=(7, 7)):

    # shape = gray.shape[::-1]
    # gray = cv2.blur(gray, (5, 5))

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # we've found the corners
        # let's refine its coordinates
        # objpoints.append(objp)
        if criteria is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # refining pixel coordinates for given 2d points.
        corners = cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), criteria)
        # pass
        # imgpoints.append(corners2)

    return ret, corners


def board_points(checkerboard):
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    return objp


def detect_boards(directory, CHECKERBOARD, show=False, wait=0, criteria=None):

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = board_points(CHECKERBOARD)

    # Extracting path of individual image stored in a given directory
    images = glob.glob(directory)
    shape = None
    for fname in images:
        print("processing", fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        shape = gray.shape[::-1]

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            img = draw_checkerboard(img, CHECKERBOARD, corners2, ret)

        if show:
            cv2.imshow('img', img)
            k = cv2.waitKey(wait)
        else:
            k = 0
        if k == ord('q'):
            break

    return shape, objpoints, imgpoints


def np_print(np_array):
    h, w = np_array.shape
    if h == 1 or w == 1:
        num_fmt = "{:.6f}"
    else:
        num_fmt = "{:.3f}"

    str_array = "[\n" + ",\n".join([
        "\t[" + ",\t".join([num_fmt.format(v).rjust(10, ' ') for v in row]) + "]"
        for row in np_array
    ]) + "\n]"
    ret = "np.array(" + str_array + ")"
    return ret

def do_calib(img_shape, obj_points, world_points):
    print("num_points", len(obj_points))
    print("calibrating...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        world_points,
        img_shape,
        None, None
    )

    np.set_printoptions(suppress=True)
    # print("Camera matrix : \n")
    # print(mtx.round(3))
    # print("dist : \n")
    # print(dist)

    print("# Intrinsic parameters")
    print("K = ", np_print( mtx ))

    print("")

    print("dist_coeffs = ", np_print(dist))

    return mtx, dist


def calib_zhang(object_points, world_points):
    import zhang

    n = len(object_points)
    first = object_points[0]
    m = first.shape[1]

    object_points = np.array(object_points).reshape(n, m, 3)
    world_points = np.array(world_points).reshape(n, m, 2)

    homographies = [zhang.compute_homography(wp, ip) for wp, ip in
                    # zip(world_points, object_points)
                    zip(object_points, world_points)
                    ]

    mint = zhang.intrinsic_from_homographies(homographies)

    extrinsics = [zhang.extrinsics_from_homography(H, mint) for H in homographies]
    for i, (R, t) in enumerate(extrinsics):
        print(f"Extrinsics for image {i + 1}:\nR:\n{R}\nt:\n{t}\n")

    return mint




if __name__ == "__main__":

    # Defining the dimensions of checkerboard
    directory = './cam3_stereo_images/calib_left_*.jpg'
    CHECKERBOARD = (10, 7)

    img_shape, obj_points, world_points = detect_boards(
        directory,
        CHECKERBOARD, show=True, wait=1
    )

    # calib using CV
    mint, dist = do_calib(img_shape, obj_points, world_points)

    # this code compares "hand made" zhang calibration method
    # # calib using zhang
    # mint2 = calib_zhang(obj_points, world_points)
    #
    # print(mint.round())
    # print(mint2.round())
