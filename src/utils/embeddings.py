from .f_extraction import feature_extractor


# def extract_embedding(
#    images: List[np.ndarray],
#    async_mode: bool = False,
# ) -> np.ndarray:
#    if not async_mode:
#        return feature_extractor.inference(images)
#
#    return feature_extractor.inference_async(images)
def extract_embedding(image, async_mode: bool = False):
    if not async_mode:
        return feature_extractor.inference(image[0])

    return feature_extractor.inference(image[0])


def check_inside(box_a: list, box_b: list) -> bool:
    """
    Check if box_a is inside box_b

    Args:
        box_a (list): [x1, y1, x2, y2]
        box_b (list): [x1, y1, x2, y2]

    The Origin is at the top-left corner,
    x-axis is the horizontal axis and y-axis is the vertical axis
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    if x1_a >= x1_b and y1_a >= y1_b and x2_a <= x2_b and y2_a <= y2_b:
        return True

    return False
