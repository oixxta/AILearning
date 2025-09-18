"""
합성곱(Convolutuion)의 이해 : filter, stride, padding
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

im = rgb2gray(data.coffee())
im = resize(im, (64, 64))
print(im.shape)             #(64, 64)

plt.axis('off')
plt.imshow(im, cmap='gray')
plt.show()                  #원본 이미지 : 커피가 담긴 머그컵

# 합성곱 필터 (3, 3)
filter1 = np.array([
    [1, 1, 1], 
    [0, 0, 0],
    [-1, -1, -1]
])

new_image = np.zeros(im.shape)      # 0으로 채워져 있는 메트릭스 생성
im_pad = np.pad(im, 1, 'constant')  # 상하좌우에 1픽셀씩 추가함, 새로 추가된 픽셀을 0으로 채움.

# 합성곱 연산
for i in range(im.shape[0]):    #세로방향
    for j in range(im.shape[1]):    #가로방향
        ii, jj = i + 1, j + 1
        try:
            new_image[i, j] = (im_pad[ii-1, jj-1] * filter1[0, 0] +
            im_pad[ii-1, jj  ] * filter1[0, 1] +
            im_pad[ii-1, jj+1] * filter1[0, 2] +
            im_pad[ii  , jj-1] * filter1[1, 0] +
            im_pad[ii  , jj  ] * filter1[1, 1] +
            im_pad[ii  , jj+1] * filter1[1, 2] +
            im_pad[ii+1, jj-1] * filter1[2, 0] +
            im_pad[ii+1, jj  ] * filter1[2, 1] +
            im_pad[ii+1, jj+1] * filter1[2, 2])
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='gray')
plt.show()