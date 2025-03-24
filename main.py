import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from image_processor import fast_denoise, fast_sharpen, fast_contrast_enhance, enhance_card_symbols
import time

class ImageDenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("B&W Image Denoising App")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.enhancement_mode = tk.StringVar(value="standard")
        
        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create and configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Create widgets
        self.create_widgets()
        
        # Configure root grid
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
    def create_widgets(self):
        # Create buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W+tk.E)
        
        # Upload button
        self.upload_btn = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Processing mode options
        mode_frame = ttk.LabelFrame(button_frame, text="Enhancement Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        # Radio buttons for enhancement mode
        ttk.Radiobutton(mode_frame, text="Standard", variable=self.enhancement_mode, 
                       value="standard").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Card Symbols", variable=self.enhancement_mode, 
                       value="card_symbols").pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_btn = ttk.Button(button_frame, text="Process Image", command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.state(['disabled'])
        
        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save Processed Image", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.state(['disabled'])
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Processing time label
        self.time_label = ttk.Label(button_frame, text="Processing time: 0 ms")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # Create image display frames
        self.original_frame = ttk.LabelFrame(self.main_frame, text="Original Image", padding="5")
        self.original_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.processed_frame = ttk.LabelFrame(self.main_frame, text="Processed Image", padding="5")
        self.processed_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Create image labels
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        # Configure frames to expand
        self.original_frame.columnconfigure(0, weight=1)
        self.original_frame.rowconfigure(0, weight=1)
        self.processed_frame.columnconfigure(0, weight=1)
        self.processed_frame.rowconfigure(0, weight=1)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                # Read and display original image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", "Could not load image")
                    return
                
                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Resize image to fit display while maintaining aspect ratio
                display_image = self.resize_image(rgb_image)
                
                # Convert to PhotoImage and display
                self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
                self.original_label.configure(image=self.original_photo)
                
                # Enable process button
                self.process_btn.state(['!disabled'])
                self.processed_label.configure(image='')
                
                # Reset time label
                self.time_label.configure(text="Processing time: 0 ms")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def resize_image(self, image, max_size=500):
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image
    
    def process_image(self):
        if self.original_image is None:
            return
        
        # Disable buttons during processing
        self.upload_btn.state(['disabled'])
        self.process_btn.state(['disabled'])
        self.save_btn.state(['disabled'])
        self.progress_var.set(0)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_image_thread)
        thread.start()
    
    def process_image_thread(self):
        try:
            start_time = time.time()
            
            # Convert to grayscale if it's a color image
            if len(self.original_image.shape) == 3:
                is_grayscale = np.allclose(self.original_image[:,:,0], self.original_image[:,:,1]) and \
                             np.allclose(self.original_image[:,:,1], self.original_image[:,:,2])
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.original_image.copy()
                is_grayscale = True
            
            self.progress_var.set(20)
            
            # Process based on selected enhancement mode
            if self.enhancement_mode.get() == "card_symbols":
                # Use specialized card symbol enhancement
                self.processed_image = enhance_card_symbols(gray)
                self.progress_var.set(90)
            else:
                # Use standard denoising pipeline
                denoised = fast_denoise(gray)
                self.progress_var.set(60)
                
                # Apply sharpening
                sharpened = fast_sharpen(denoised)
                self.progress_var.set(80)
                
                # Apply contrast enhancement
                self.processed_image = fast_contrast_enhance(sharpened)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Convert to RGB for display
            display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            
            # Resize for display
            display_image = self.resize_image(display_image)
            
            # Update GUI in main thread
            self.root.after(0, self.update_processed_display, display_image, processing_time)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
        finally:
            self.root.after(0, self.processing_complete)
    
    def update_processed_display(self, display_image, processing_time):
        self.processed_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
        self.processed_label.configure(image=self.processed_photo)
        self.time_label.configure(text=f"Processing time: {processing_time:.1f} ms")
    
    def processing_complete(self):
        self.progress_var.set(100)
        self.upload_btn.state(['!disabled'])
        self.process_btn.state(['!disabled'])
        self.save_btn.state(['!disabled'])
    
    def save_image(self):
        if self.processed_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

def main():
    root = tk.Tk()
    app = ImageDenoisingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
