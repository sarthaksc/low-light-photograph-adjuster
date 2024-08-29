import cv2
import numpy as np

def solution(image_path_a, image_path_b):

    def bilateral_filter(image1, image2, diameter, sigma_color, sigma_space):
        height, width, _ = image1.shape
        result = np.zeros_like(image1, dtype=np.float64)
        for i in range(height):
            for j in range(width):
                intensity = image2[i, j]
                x, y = np.meshgrid(
                    np.arange(max(0, i - diameter), min(height, i + diameter + 1)),
                    np.arange(max(0, j - diameter), min(width, j + diameter + 1))
                )
                spat_dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                intensity_diff = intensity - image2[x, y]
                spat_wt = (1 / (2 * np.pi * sigma_space ** 2)) * np.exp(-spat_dist ** 2 / (2 * sigma_space ** 2))
                intensity_wt = (1 / (2 * np.pi * sigma_color ** 2)) * np.exp(-intensity_diff ** 2 / (2 * sigma_color ** 2))
                wt = spat_wt * intensity_wt
                weighted_sum = np.sum(image1[x, y] * wt[:, :, np.newaxis], axis=(0, 1))
                total_weight = np.sum(wt, axis=(0, 1))
                result[i, j] = weighted_sum / total_weight
        return result.astype(np.uint8)

    no_flash=cv2.imread(image_path_a)
    gray_no_flash = cv2.cvtColor(no_flash, cv2.COLOR_BGR2LAB)
    intensity_no_flash = gray_no_flash[:, :, 0]

    flash=cv2.imread(image_path_b)
    gray_flash = cv2.cvtColor(flash, cv2.COLOR_BGR2LAB)
    intensity_flash = gray_flash[:, :, 0]    

    intensity_flash = intensity_flash * (np.sum(intensity_no_flash)/np.sum(intensity_flash))
    filtered = bilateral_filter(no_flash,intensity_flash,15,8,7)
    filtered_lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    filtered_intensity = filtered_lab[:, :, 0]
    filtered_color = filtered / filtered_intensity[:, :, np.newaxis]
    filtered_color *= 255

    filtered_intensity_norm = intensity_flash.astype(float) / 255.0
    merged = filtered_color * filtered_intensity_norm[:, :, np.newaxis]
    merged= np.clip(merged, 0, 255).astype(np.uint8)

    intensity_difference = filtered_intensity - intensity_flash
    _, binary_intensity_difference = cv2.threshold(intensity_difference,0,255,cv2.THRESH_BINARY)
    binary_intensity_difference_inv = (cv2.bitwise_not(binary_intensity_difference.astype(np.uint8)))/255
    filtered_mask = filtered*binary_intensity_difference_inv.astype(np.uint8)[...,None]
    merged_mask = merged*(binary_intensity_difference/255).astype(np.uint8)[...,None]
    result = cv2.addWeighted(merged_mask, 1.3, filtered_mask, 1, 0)

    return result