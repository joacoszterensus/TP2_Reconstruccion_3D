import numpy as np
import cv2


def create_charuco_board(
        squares_x=5,
        squares_y=7,
        square_length=0.04,
        marker_length=0.02,
        dictionary_type=cv2.aruco.DICT_6X6_250
):
    """
    Crea un tablero Charuco con los parámetros especificados.

    Args:
        squares_x: Número de cuadrados en el eje X
        squares_y: Número de cuadrados en el eje Y
        square_length: Longitud física de los cuadrados (en metros)
        marker_length: Longitud física de los marcadores (en metros)
        dictionary_type: Tipo de diccionario ArUco (por defecto DICT_6X6_250)

    Returns:
        Objeto CharucoBoard de OpenCV
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    return board


def detect_aruco_markers(image, aruco_dict, detector_params=None):

    """
    Detecta marcadores Aruco en una imagen

    Args:
        image: Imagen de entrada (BGR o escala de grises)
        detector_params: Parámetros opcionales para el detector

    Returns:
        dict: {
            'corners': posiciones de esquinas (np.array),
            'ids': identificadores (np.array),
            'rejected': marcadores rechazados,
            'image': imagen con resultados (si draw_results=True)
        }
        o None si no se detectan marcadores
    """

    # convierte a escala de grises si es necesario
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if not detector_params:
        detector_params = cv2.aruco.DetectorParameters()

    # crea un detector
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        detector_params
    )

    # detecta los marcadores
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # empaqueta los resultados
    result = {
        'corners': corners,
        'ids': ids,
        'rejected': rejected
    }

    return result


def detect_charuco_markers(image, board, detector_params=None):
    """
    Detecta marcadores Charuco en una imagen dado un tablero.

    Args:
        image: Imagen de entrada (BGR o escala de grises)
        board: Objeto CharucoBoard previamente creado
        detector_params: Parámetros opcionales para el detector

    Returns:
        dict: {
            'corners': posiciones de esquinas (np.array),
            'ids': identificadores (np.array),
            'rejected': marcadores rechazados,
            'image': imagen con resultados (si draw_results=True)
        }
        o None si no se detectan marcadores
    """
    # convierte a escala de grises si es necesario
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if not detector_params:
        detector_params = cv2.aruco.DetectorParameters()

    # crea un detector
    detector = cv2.aruco.ArucoDetector(
        board.getDictionary(),
        detector_params
    )

    # detecta los marcadores
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # empaqueta los resultados
    result = {
        'corners': corners,
        'ids': ids,
        'rejected': rejected
    }

    return result


def draw_charuco_markers(image, detection_result, draw_rejected=False):
    """
    Dibuja los resultados de la detección sobre la imagen.

    Args:
        image: Imagen original (BGR)
        detection_result: Tupla (corners, ids, rejected) de detect_charuco_markers
        draw_rejected: Si True, marca los candidatos rechazados

    Returns:
        Imagen con anotaciones (BGR)
    """
    output_image = image.copy()

    if detection_result is None:
        return output_image

    corners = detection_result["corners"]
    ids = detection_result["ids"]
    rejected = detection_result["rejected"]

    # dibuja marcadores aceptados (verdes)
    if ids is not None and len(ids) > 0:
        output_image = cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

    # dibuja rechazados (rojos) si se solicita
    if draw_rejected and rejected is not None and len(rejected) > 0:
        cv2.aruco.drawDetectedMarkers(
            output_image,
            rejected,
            borderColor=(0, 0, 255)  # Rojo para rechazados
        )

    return output_image


def draw_aruco_results(image, detection_result, size=5):
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    if detection_result is None:
        return result

    corners = detection_result["corners"]
    ids = detection_result["ids"]
    rejected = detection_result["rejected"]

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(result, topLeft, topRight, (0, 0, 255), size)
            cv2.line(result, topRight, bottomRight, (0, 255, 0), size)
            cv2.line(result, bottomRight, bottomLeft, (255, 0, 0), size)
            cv2.line(result, bottomLeft, topLeft, (0, 255, 255), size)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(result, (cX, cY), 3 * size, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image

            org = (cX + 15, cY + 30)
            if org[1] < 0:
                org = (org[0], 0)
            font_scale = size
            cv2.putText(
                img=result,
                text=str(markerID),
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=size+10
            )
            cv2.putText(
                img=result,
                text=str(markerID),
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 255, 0),
                thickness=size+3
            )

            # print("[INFO] ArUco marker ID: {}".format(markerID))

    return result


def get_marker_corners_3d(board):
    """Precomputes 3D corners for all markers in the board"""
    marker_length = board.getMarkerLength()
    half_size = marker_length / 2
    marker_corners = []

    corners3d = board.getChessboardCorners()

    for center in corners3d:
        corners = np.array([
            center + [-half_size, -half_size, 0],  # Top-left
            center + [half_size, -half_size, 0],  # Top-right
            center + [half_size, half_size, 0],  # Bottom-right
            center + [-half_size, half_size, 0]  # Bottom-left
        ], dtype=np.float32)
        marker_corners.append(corners)

    return marker_corners


def estimate_camera_pose_with_homography(
        image,
        board,
        detection,
        calibration,
        undistort=False
):
    corners = detection['corners']
    ids = detection['ids']

    K, dist = calibration

    if len(ids) < 4:
        return None  # for estimating homography we need at least 4 points

    # builds 2D correspondences (unage) <-> 2D (board plane)
    image_points = []
    board_points = []

    board_ids = board.getIds()
    points3d = board.getObjPoints()
    for corners, id in zip(corners, ids):
        id = int(id)
        if id not in board_ids:
            continue

        # gets the corners in the boards coordinates
        point3d = points3d[id]  # (4, 3)
        #obj_points_2d = [pt[:2] for pt in obj_corners]  # saca Z

        image_points.extend(corners[0])  # (4, 2)
        board_points.extend(point3d)  # (4, 2)

    image_points = np.array(image_points, dtype=np.float32)
    board_points = np.array(board_points, dtype=np.float32)

    if undistort:
        use_image_points = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = image_points

    # estimar homografía
    H, inliers = cv2.findHomography(
        board_points[:, :2],
        use_image_points,
        # method=cv2.RANSAC
        method=cv2.LMEDS
    )

    if H is None:
        return None

    # gets all the charuco's corners (X, Y) in the board plane
    charuco_obj_points = []
    charuco_ids = []
    chess = board.getChessboardCorners()
    for i in range(chess.shape[0]):
        corner = chess[i]  # (3,)
        charuco_obj_points.append(corner)
        charuco_ids.append(i)

    charuco_board_points = np.array([p[:2] for p in charuco_obj_points], dtype=np.float32).reshape(-1, 1, 2)

    # projects the points using the found homography
    projected_charuco_corners = cv2.perspectiveTransform(charuco_board_points, H)
    projected_charuco_corners = projected_charuco_corners.reshape(-1, 2)

    obj_pts = np.array(charuco_obj_points, dtype=np.float32)
    # img_pts = projected_charuco_corners

    if undistort:
        use_image_points = cv2.undistortPoints(
            projected_charuco_corners.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = projected_charuco_corners

    # solves the pose using PnP. Since all the points are on a plane we use IPPE
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        use_image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    return success, rvec, tvec


if __name__ == '__main__':
    board = create_charuco_board()
    print(board)
