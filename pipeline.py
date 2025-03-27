import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageClarityPipeline:
    """
    A pipeline for enhancing the clarity of images with symbols/shapes.
    Specifically designed to transform blurry images into clear ones.
    """
    
    def __init__(self, input_path=None):
        """
        Initialize the pipeline with optional input path.
        
        Args:
            input_path (str, optional): Path to the input image. Defaults to None.
        """
        self.image = None
        # Initialize the processed_images dictionary
        self.processed_images = {}
        if input_path:
            self.load_image(input_path)
    
    def load_image(self, path):
        """
        Load an image from the given path.
        
        Args:
            path (str): Path to the image file
            
        Returns:
            self: For method chaining
        """
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Could not load image from {path}")
        
        # Convert BGR to RGB for visualization purposes
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.processed_images["original"] = self.image.copy()
        return self
    
    def convert_to_grayscale(self):
        """
        Convert the image to grayscale.
        
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.processed_images["grayscale"] = self.image.copy()
        return self
    
    def apply_denoising(self, strength=10):
        """
        Apply non-local means denoising to reduce noise while preserving edges.
        
        Args:
            strength (int): Strength of the denoising operation.
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # For grayscale images
        if len(self.image.shape) == 2:
            self.image = cv2.fastNlMeansDenoising(self.image, None, strength, 7, 21)
        # For color images
        else:
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, strength, 10, 7, 21)
            
        self.processed_images["denoised"] = self.image.copy()
        return self
        
    def enhance_contrast(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Enhance the contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (tuple): Size of grid for histogram equalization
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Ensure image is grayscale
        if len(self.image.shape) > 2:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            img = self.image
            
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.image = clahe.apply(img)
        self.processed_images["contrast_enhanced"] = self.image.copy()
        return self
        
    def sharpen_image(self, kernel_size=3, alpha=1.5):
        """
        Sharpen the image using unsharp masking.
        
        Args:
            kernel_size (int): Size of the Gaussian kernel for blurring
            alpha (float): Weight of the sharpening effect
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Gaussian blur
        gaussian = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        
        # Unsharp masking
        self.image = cv2.addWeighted(self.image, 1 + alpha, gaussian, -alpha, 0)
        self.processed_images["sharpened"] = self.image.copy()
        return self
        
    def apply_adaptive_threshold(self, block_size=11, C=2):
        """
        Apply adaptive thresholding to better separate shapes from background.
        
        Args:
            block_size (int): Size of pixel neighborhood used for thresholding
            C (int): Constant subtracted from the mean
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.image.shape) > 2:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            img = self.image
            
        self.image = cv2.adaptiveThreshold(
            img, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            C
        )
        self.processed_images["adaptive_threshold"] = self.image.copy()
        return self
        
    def remove_noise(self, kernel_size=3):
        """
        Remove small noise using morphological operations.
        
        Args:
            kernel_size (int): Size of the kernel for morphological operations
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening operation (erosion followed by dilation)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        self.processed_images["noise_removed"] = self.image.copy()
        return self
        
    def edge_enhancement(self):
        """
        Enhance edges using the Canny edge detector and combine with original.
        
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.image.shape) > 2:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            img = self.image.copy()
            
        # Apply Canny edge detection
        edges = cv2.Canny(img, 100, 200)
        
        # Dilate edges to make them more visible
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine original with edges
        if len(self.image.shape) > 2:
            # For color images
            edge_image = self.image.copy()
            edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            edge_image[edge_mask > 0] = [255, 255, 255]  # Highlight edges in white
            self.image = edge_image
        else:
            # For grayscale images
            self.image = cv2.bitwise_and(self.image, self.image, mask=edges)
            
        self.processed_images["edge_enhanced"] = self.image.copy()
        return self
        
    def fix_perspective(self, target_size=(500, 500)):
        """
        Fix perspective distortion if required.
        
        Args:
            target_size (tuple): Target size for the rectified image
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # This is a simplified implementation - in a real-world application,
        # you would detect the contour of the grid and transform it
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if len(self.image.shape) > 2 else self.image.copy()
        
        # Apply threshold or edge detection
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the grid)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to get the corners
            epsilon = 0.1 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:  # If we have a quadrilateral
                # Define the destination points
                dst_pts = np.array([
                    [0, 0],
                    [target_size[0], 0],
                    [target_size[0], target_size[1]],
                    [0, target_size[1]]
                ], dtype=np.float32)
                
                # Get the corner points from the approximated contour
                src_pts = np.array([approx[i][0] for i in range(4)], dtype=np.float32)
                
                # Order the points correctly
                # This is a simplified approach - for a robust solution, sort points properly
                
                # Get the transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Apply the transformation
                self.image = cv2.warpPerspective(self.image, M, target_size)
                self.processed_images["perspective_fixed"] = self.image.copy()
        
        return self
        
    def save_image(self, output_path):
        """
        Save the processed image to the given path.
        
        Args:
            output_path (str): Path to save the image
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image to save")
        
        # Convert to BGR for saving with OpenCV
        if len(self.image.shape) == 3:
            save_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        else:
            save_img = self.image
            
        cv2.imwrite(output_path, save_img)
        return self
        
    def visualize_pipeline(self, figsize=(15, 10)):
        """
        Visualize all steps in the pipeline.
        
        Args:
            figsize (tuple): Figure size for the plot
            
        Returns:
            None
        """
        if not self.processed_images:
            print("No processed images to visualize")
            return
            
        n = len(self.processed_images)
        cols = 3
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=figsize)
        
        for i, (name, img) in enumerate(self.processed_images.items()):
            plt.subplot(rows, cols, i + 1)
            
            if len(img.shape) == 2 or img.shape[2] == 1:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
                
            plt.title(name)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    def smooth_boundaries(self, kernel_size=3, sigma=0.8):
        """
        Apply Gaussian blur to smooth symbol boundaries.
        
        Args:
            kernel_size (int): Size of the Gaussian kernel
            sigma (float): Standard deviation for Gaussian kernel
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
        self.processed_images["smoothed"] = self.image.copy()
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        Args:
            d (int): Diameter of each pixel neighborhood
            sigma_color (float): Filter sigma in the color space
            sigma_space (float): Filter sigma in the coordinate space
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        self.processed_images["bilateral_filtered"] = self.image.copy()
        return self
        
    def apply_local_binary_pattern(self):
        """
        Apply Local Binary Pattern for texture enhancement.
        
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Ensure grayscale
        if len(self.image.shape) > 2:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            img = self.image.copy()
            
        radius = 1
        n_points = 8 * radius
        
        # Calculate LBP
        lbp = np.zeros_like(img)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                center = img[i, j]
                pattern = 0
                for k in range(8):
                    row = i + int(round(radius * np.cos(k * 2 * np.pi / 8)))
                    col = j + int(round(radius * np.sin(k * 2 * np.pi / 8)))
                    pattern |= (img[row, col] > center) << k
                lbp[i, j] = pattern
                
        self.image = lbp
        self.processed_images["lbp_enhanced"] = self.image.copy()
        return self
        
    def apply_morphological_gradient(self, kernel_size=3):
        """
        Apply morphological gradient for edge enhancement.
        
        Args:
            kernel_size (int): Size of the kernel for morphological operations
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Morphological gradient = dilation - erosion
        dilation = cv2.dilate(self.image, kernel, iterations=1)
        erosion = cv2.erode(self.image, kernel, iterations=1)
        self.image = cv2.subtract(dilation, erosion)
        
        self.processed_images["morph_gradient"] = self.image.copy()
        return self

    def apply_simple_threshold(self, threshold=127):
        """
        Apply simple binary thresholding.
        
        Args:
            threshold (int): Threshold value
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        _, self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        self.processed_images["simple_threshold"] = self.image.copy()
        return self

    def apply_dilation(self, kernel_size=3, iterations=1):
        """
        Apply dilation to expand white regions.
        
        Args:
            kernel_size (int): Size of the kernel
            iterations (int): Number of times to apply dilation
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=iterations)
        self.processed_images["dilated"] = self.image.copy()
        return self

    def remove_salt_noise(self, kernel_size=3):
        """
        Remove salt noise using median filtering.
        Median filtering is particularly effective for salt and pepper noise.
        
        Args:
            kernel_size (int): Size of the median filter kernel
            
        Returns:
            self: For method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Apply median filter
        self.image = cv2.medianBlur(self.image, kernel_size)
        self.processed_images["salt_noise_removed"] = self.image.copy()
        return self


if __name__ == "__main__":

    # Example usage:
    pipeline = ImageClarityPipeline("raw.jpg")
    (pipeline
        .convert_to_grayscale()
        # Initial contrast enhancement
        .enhance_contrast(clip_limit=3.0, tile_grid_size=(3, 3))
        # Apply salt noise removal with median filter
        .remove_salt_noise(kernel_size=5)
        # Apply regular denoising for remaining noise
        .apply_denoising(strength=10)
        # Apply light sharpening to restore details
        .sharpen_image(kernel_size=3, alpha=1.2)
        # Final cleanup with opening operation to remove any remaining specks
        .remove_noise(kernel_size=5)
        # Save the result
        .save_image("processed.jpg"))

    # Visualize all steps
    pipeline.visualize_pipeline(figsize=(20, 12))