import cv2
import matplotlib
import torch
from PIL import Image
from matplotlib import cm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> Image.Image:
    '''
        Adopted from https://github.com/frgfm/torch-cam
    '''
    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def plot_activation(mask_np, x, image_size, score, class_name, tempeature=0.5):
    # normalization to make presentation clearer
    mask_np = mask_np - mask_np.min()
    mask_np = mask_np / mask_np.max()
    mask_np = np.power(mask_np, tempeature)

    mask_resize = cv2.resize(mask_np, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    mask_resize = np.expand_dims(mask_resize, axis=2)

    from torchvision.transforms.functional import to_pil_image
    result = overlay_mask(to_pil_image(x[0].cpu().float()), to_pil_image(mask_np, mode='F'), alpha=0.4)
    im = np.transpose(x[0].cpu().float().numpy(), (1, 2, 0))

    ## plot
    # plt.figure(ind)
    plt.subplot(1, 4, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.title('Activation Map, Class: {}'.format(class_name))
    plt.subplot(1, 4, 3)
    plt.imshow(result)
    plt.axis('off')
    plt.title('Inference Score {:.2f}'.format(score))
    #
    plt.subplot(1, 4, 4)
    plt.imshow(im * mask_resize)
    plt.axis('off')
