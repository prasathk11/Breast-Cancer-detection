import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def _segment_maxima(image: np.ndarray,
                    low: int=1, high: int=250, debug: bool=False) -> np.ndarray:
    """
    Parameters:
    -----------
    img, numpy.ndarray
        Ground-truth image with mask outline
    low, int
        Lower bound threshold to segment
    high, int
        Upper bound threshold to segment
    debug, bool
        Wether to display segmentation mask for debugging purposes.
    Returns:
    --------
    numpy.ndarray
        Binary mask with segmented colors.
    """

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.logical_and(image > low, image < high)*255
    image = image.astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    image = cv2.erode(image,kernel,iterations=2)
    image = cv2.dilate(image,kernel,iterations=2)
    h, w = image.shape
    cv2.rectangle(image,(0,0),(w,h),0,thickness=10)
    if debug:
        plt.imshow(image)

    return image


def _segment_colors(image: np.ndarray) -> np.ndarray:
    """
    Parameters:
    -----------
    img, numpy.ndarray
        Ground-truth image with mask outline
    Returns:
    --------
    numpy.ndarray
        Binary mask with segmented colors.
    """
    hsv_bound_map = {
        'green': [(35, 20, 20), (75, 255, 255)],
        'orange': [(7, 20, 20), (15, 255, 255)],
        'yellow': [(10, 20, 20), (50, 255, 255)]
    }
    
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # binary mask for each color
    masks = [cv2.inRange(image, hsv_bound_map[color][0], hsv_bound_map[color][1])
             for color in hsv_bound_map.keys()]
    # get intersection of masks
    masks = np.array(masks).any(axis=0).astype(np.uint8)

    return masks


def _segment_white(image: np.ndarray, high: int=210, low: int=230) -> np.ndarray:

    mask = 2.0 * (image > high) * (image < low)
    mask = (mask.sum(axis=2) > 1).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.dilate(mask, kernel, iterations=3)

    return mask


def _find_largest_box(binary_image: np.ndarray) -> Tuple[int]:
    """
    Parameters:
    -----------
    binary_image, numpy.ndarray
        Binary segmentation mask.
    Returns:
    --------
    Tuple(int)
        xmin, ymin, width, height coordinates of rectangular box containing
        largest connected object in the image. 
    """

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return None

    largest_contour = None
    max_area = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    box = cv2.boundingRect(largest_contour)

    return box


def crop_image(image: np.ndarray,
               xmin: int, ymin: int, width: int, height: int) -> np.ndarray:
    """
    Parameters:
    -----------
    image, numpy.ndarray
        Binary segmentation mask.
    xmin, int
        Lower corner along width dimension of rectangular box to crop image
    ymin, int
        Lower corner along height dimension of rectangular box to crop image
    width, int
        Width of rectangular box to crop image
    height, int
        Height of rectangular box to crop image
    Returns:
    --------
    numpy.ndarray
        Cropped image.
    """

    return image[ymin:ymin+height, xmin:xmin+width]


def remove_borders(image: np.ndarray) -> np.ndarray:
    """ Remove black/white borders from image. Expects image as np.ndarray.
    
    Parameters:
    -----------
    image, numpy.ndarray
        Image to be preprocessed.
    Returns:
    --------
    numpy.ndarray
        Image with black and white border removed.
    """
    binary = _segment_maxima(image)
    largest_object_box = _find_largest_box(binary)
    cropped_image = crop_image(image, *largest_object_box)
            
    return cropped_image


def resize(image: np.ndarray, *args, **kwargs) -> np.ndarray:
    """ Resize image as np.ndarray.
    
    Parameters:
    -----------
    image, numpy.ndarray
        Image to be resized
    *args
        Positional args to PIL.Image.resize() function
    *kwargs
        Keyword args to PIL.Image.resize() function
    Returns:
    --------
    numpy.ndarray
        Resized image.
    """

    image = image.copy().astype(np.uint8)

    # resize as PIL
    pil = Image.fromarray(image)
    pil = pil.resize(*args, **kwargs)

    #back to array
    image = np.array(pil)

    return image


def inpaint(image: np.ndarray) -> np.ndarray:
    image = image.copy()

    for segmentor in [_segment_colors, _segment_white]:
        mask = segmentor(image)
        mask = cv2.UMat(mask)

        # inpainted cv2::UMat
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # back to numpy.ndarray
        image = image.get()

    return image


def preprocess(image):
    """    
    Parameters:
    -----------
    image, numpy.ndarray
        Image to be preprocessed
    Returns:
    --------
    numpy.ndarray
        Preprocessed image.
    """

    image = remove_borders(image)
    image = inpaint(image)
    # when we do resize some data is lost, so can be done based on the model
    #image = resize(image, (256, 256))

    return image

def cast_to_rgb(image: np.ndarray) -> np.ndarray:
    """    
    Given a valid uint8 image convert it to rgb
    """
    if image.ndim!=2 and image.ndim!=3 or image.dtype!='uint8':
        raise Exception("Image format not supported")

    if image.ndim==2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    if image.shape[-1]!=3:
        image = cv2.cvtColor(image[...,0],cv2.COLOR_GRAY2RGB)

    return image




