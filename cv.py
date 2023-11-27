# Copyright 2023 a1147
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import cv2


img = cv2.imread('68.png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([180, 30, 255], dtype=np.uint8)
mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
# kernel = np.ones((5, 5), np.uint8)
# mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
# mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
cv2.imshow('origin', img)
cv2.imshow('test', mask_white)
cv2.waitKey(0)