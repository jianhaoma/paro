import numpy as np

def calculate_quantization_ratio(nums, tolerance=1e-2):
    '''Return the ratio of numbers that are close to integers within tolerance'''
    ratio = np.isclose(nums, np.round(nums), atol=tolerance).mean()
    return ratio
