import numpy as np
import cv2

from utils import plot_solution

#%% Read data
plot_solution(np.load("data/1000/2032c_U(55.32,2.83)_dpi(128,128).npz"))
plot_solution(np.load("data/10000/2032c_U(12.15,3.62)_dpi(128,128).npz"))
plot_solution(np.load("data/10000/2032c_U(34.13,0.00)_dpi(128,128).npz"))
plot_solution(np.load("data/10000/ag04_U(42.42,3.43)_dpi(128,128).npz"))

sol = np.load("data/1000_unstruct/ag09_U(58.47,4.61).npz")
sol["variables"]

#%% Basic open cv inpainting
solution = np.load("data/10000/2032c_U(12.15,3.62)_dpi(128,128).npz")
image = np.concatenate(
    [
        solution["Vx"] / np.linalg.norm(solution["Vx"]),
        solution["Vy"] / np.linalg.norm(solution["Vy"]),
        solution["Pressure"] / np.linalg.norm(solution["Pressure"]),
    ],
    axis=-1,
)
x_crop = np.random.randint(0, 64)
y_crop = np.random.randint(0, 64)
tile = image[y_crop : y_crop + 64, x_crop : x_crop + 64]
mask = np.ones((64, 64, 64))
mask[2:62, 2:62, :] = 0
dst = cv2.inpaint(tile, mask, 3, cv2.INPAINT_TELEA)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
