import numpy as np
from PIL import Image


def histogram_from_grayscale(img: Image) -> np.array:

    histogram = np.zeros(256)
    # Reshape para facilitar iteração
    img_array = np.asarray(img).flatten()

    histogram, _ = np.histogram(img_array, 256, [0, 256])

    return histogram


def histogram_from_rgb(img: Image) -> np.array:

    r_channel = img.getchannel(0)
    g_channel = img.getchannel(1)
    b_channel = img.getchannel(2)

    r_hist = histogram_from_grayscale(r_channel)
    g_hist = histogram_from_grayscale(g_channel)
    b_hist = histogram_from_grayscale(b_channel)

    return np.concatenate([r_hist, g_hist, b_hist])


def luminance_gray_scale(img: Image) -> Image:

    img_array = np.asarray(img)
    output_array = np.zeros(shape=img_array.shape[:2])

    # Razão de contribuição dos canais: 0.21R + 0.71G + 0.07B
    output_array = 0.21 * img_array[:,:,0] + 0.71 * img_array[:,:,1] + 0.07 * img_array[:,:,2]

    return Image.fromarray(np.uint8(output_array))


def get_pdf(img: Image, rgb: bool = False) -> np.array:

    if rgb:
        img_histogram = histogram_from_rgb(img)
    else:
        img_histogram = histogram_from_grayscale(img)

    n_pixels = img.size[0] * img.size[1]
    img_pdf = (1/n_pixels) * img_histogram

    return img_pdf