# cython: boundscheck=False, wraparound=False, language_level=3
import numpy as np
cimport numpy as np
import cv2

def fast_denoise(np.ndarray[np.uint8_t, ndim=2] gray):
    cdef:
        np.ndarray[np.uint8_t, ndim=2] denoised_nlm
        np.ndarray[np.uint8_t, ndim=2] denoised_bilateral
        np.ndarray[np.uint8_t, ndim=2] edges
        np.ndarray[np.float64_t, ndim=2] weight_mask
        np.ndarray[np.uint8_t, ndim=2] result
        int i, j
        double weight
    
    # Fast non-local means denoising
    denoised_nlm = cv2.fastNlMeansDenoising(gray, None, h=9, templateWindowSize=7, searchWindowSize=21)
    
    # Fast bilateral filtering
    denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Fast edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Create weight mask using numpy operations
    weight_mask = np.where(edges > 0, 0.7, 0.0)
    
    # Fast blending using numpy operations
    result = (gray * weight_mask + cv2.addWeighted(denoised_nlm, 0.6, denoised_bilateral, 0.4, 0) * (1 - weight_mask)).astype(np.uint8)
    
    return result

def fast_sharpen(np.ndarray[np.uint8_t, ndim=2] img):
    cdef:
        np.ndarray[np.float64_t, ndim=2] kernel
        np.ndarray[np.uint8_t, ndim=2] result
    
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]], dtype=np.float64)
    
    result = cv2.filter2D(img, -1, kernel)
    return result

def fast_contrast_enhance(np.ndarray[np.uint8_t, ndim=2] img):
    cdef:
        np.ndarray[np.uint8_t, ndim=2] result
    
    result = cv2.convertScaleAbs(img, alpha=1.1, beta=5)
    return result

# Improved implementation for card symbol enhancement
def enhance_card_symbols(np.ndarray[np.uint8_t, ndim=2] gray_input):
    """
    Enhanced processing targeting card symbols while keeping the background clean.
    Uses a gentler approach that preserves more detail.
    """
    # First, make a copy to avoid modifying the input
    cdef np.ndarray[np.uint8_t, ndim=2] gray = gray_input.copy()
    
    # Step 1: Gentle denoising to preserve details
    cdef np.ndarray[np.uint8_t, ndim=2] denoised = cv2.fastNlMeansDenoising(
        gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Bilateral filter to preserve edges while reducing noise
    cdef np.ndarray[np.uint8_t, ndim=2] bilateral = cv2.bilateralFilter(denoised, 5, 50, 50)
    
    # Step 3: Create a cleaner base image with moderate contrast enhancement
    cdef np.ndarray[np.uint8_t, ndim=2] enhanced_base = cv2.convertScaleAbs(bilateral, alpha=1.2, beta=5)
    
    # Step 4: Use adaptive thresholding to find potential text/symbol areas
    # This is more precise than global thresholding
    cdef np.ndarray[np.uint8_t, ndim=2] adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Step 5: Clean up the mask - remove noise and small artifacts
    cdef np.ndarray[np.uint8_t, ndim=2] kernel = np.ones((2, 2), np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] symbols_mask = cv2.morphologyEx(
        adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 6: Create a cleaner edge mask to guide enhancement
    cdef np.ndarray[np.uint8_t, ndim=2] edges = cv2.Canny(bilateral, 30, 100)
    # Dilate edges slightly to ensure they're captured
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # Step 7: Combine the symbol mask with edge information for a better target
    cdef np.ndarray[np.uint8_t, ndim=2] combined_mask = cv2.bitwise_or(symbols_mask, edges)
    combined_mask = cv2.dilate(combined_mask, np.ones((2, 2), np.uint8), iterations=1)
    
    # Step 8: Prepare a more strongly enhanced version for symbols/text only
    cdef np.ndarray[np.uint8_t, ndim=2] symbol_enhanced = cv2.convertScaleAbs(bilateral, alpha=1.5, beta=15)
    
    # Apply unsharp masking for better symbol clarity
    cdef np.ndarray[np.uint8_t, ndim=2] blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
    cdef np.ndarray[np.uint8_t, ndim=2] unsharp_mask = cv2.addWeighted(bilateral, 1.5, blurred, -0.5, 0)
    
    # Step 9: Create a soft gradient mask for smooth blending
    # Convert mask to float32 for processing
    cdef np.ndarray soft_mask = combined_mask.astype(np.float32) / 255.0
    # Apply gaussian blur to create soft transitions
    soft_mask = cv2.GaussianBlur(soft_mask, (9, 9), 0)
    
    # Step 10: Blend the enhanced symbols with the base image using the soft mask
    # We'll do this with a safe operation to maintain uint8 type
    # Extract foreground (enhanced symbols) and background (clean base image)
    # using OpenCV's safe blending functions
    
    # Convert soft mask back to uint8 at higher precision (multiply by 255)
    cdef np.ndarray[np.uint8_t, ndim=2] blend_mask = (soft_mask * 255).astype(np.uint8)
    
    # Create the result with a weighted combination
    cdef np.ndarray[np.uint8_t, ndim=2] result = np.zeros_like(gray, dtype=np.uint8)
    
    # Use addWeighted with the unsharp mask and the base image
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            weight = soft_mask[i, j]
            result[i, j] = <np.uint8_t>(
                <int>(unsharp_mask[i, j] * weight + enhanced_base[i, j] * (1.0 - weight))
            )
    
    # Step 11: Apply a gentle final sharpening
    cdef np.ndarray[np.float64_t, ndim=2] sharpen_kernel = np.array([
        [-0.3, -0.3, -0.3], 
        [-0.3,  3.4, -0.3],
        [-0.3, -0.3, -0.3]
    ], dtype=np.float64)
    
    result = cv2.filter2D(result, -1, sharpen_kernel)
    
    return result 