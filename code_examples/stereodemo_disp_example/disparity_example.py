import cv2
import json
import numpy as np
from pathlib import Path

from stereodemo.method_cre_stereo import CREStereo
from stereodemo.method_opencv_bm import StereoBM, StereoSGBM
from stereodemo.methods import Calibration, InputPair, Config

models_path = Path.home() / ".cache" / "stereodemo" / "models"


calibration_file = "recursos/tune_disparity/stereodemo_calibration.json"
with open(calibration_file, "r") as f:
    calibration = Calibration.from_json(f.read())

left_image = cv2.imread("recursos/tune_disparity/left_10.jpg")
right_image = cv2.imread("recursos/tune_disparity/right_10.jpg")

pair = InputPair(left_image, right_image, calibration, "status?")
config = Config(models_path=models_path)

method = CREStereo(config)
# method = StereoBM(config)
# medhod = StereoSGBM(config)
disparity = method.compute_disparity(pair)


# np.savez("disparity.npz", disparity.disparity_pixels)
# cv2.imwrite("disparity.png", disparity.disparity_pixels)

dvis = disparity.disparity_pixels.copy()
dvis = 255 * (dvis - dvis.min()) / (dvis.max() - dvis.min())
dvis = dvis.astype('uint8')
cv2.imshow("disparity", dvis)

input = np.hstack((left_image, right_image))
cv2.imshow("input", input)
cv2.waitKey()