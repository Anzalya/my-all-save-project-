import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Загружаем изображение в градациях серого
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)

if img is None:
    print("Ошибка: изображение не найдено!")
    exit()

# Canny edge detection
edges = cv.Canny(img, 100, 200)

# Другой эффект — размытие (Gaussian Blur)
blur = cv.GaussianBlur(img, (7, 7), 0)

# Отображение
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Оригинал')
plt.axis('off')

plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge')
plt.axis('off')

plt.subplot(133)
plt.imshow(blur, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.show()
