import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from image_processor import fast_denoise, fast_sharpen, fast_contrast_enhance, enhance_card_symbols, enhance_resolution
import time
import keyboard
import win32gui
import win32con
import win32process
import win32ui
import psutil
import ctypes
from ctypes import windll, byref, create_unicode_buffer, create_string_buffer, wintypes
from ctypes.wintypes import BOOL, HWND, RECT
from functools import partial
import re
import traceback

# Add more detailed Windows API access for PrintWindow
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)

# Define required Windows API constants and functions
PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 0x00000002  # Windows 10 1703 or later

# Define Windows API function signatures
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

# DPI awareness constants
PROCESS_SYSTEM_DPI_AWARE = 1
PROCESS_PER_MONITOR_DPI_AWARE = 2

# Language support - add traditional Chinese
SUPPORTED_LANGUAGES = ["zh", "zh_TW"]

# Translations dictionary
TRANSLATIONS = {
    "zh": {
        "app_title": "高级图像处理器",
        "original_image": "原始图像",
        "processed_image": "处理后图像",
        "image_source": "图像来源",
        "upload_image": "上传图像",
        "select_app": "选择应用程序",
        "capturing_space": "捕获：跟踪时按空格键",
        "enhancement_options": "增强选项",
        "mode": "模式：",
        "standard": "标准",
        "card_symbols": "卡片符号",
        "resolution_scale": "分辨率缩放：",
        "method": "方法：",
        "downscale_speed": "缩小以提高速度",
        "display_options": "显示选项",
        "preview_size": "预览大小：",
        "small": "小",
        "medium": "中",
        "large": "大",
        "zoom": "缩放：",
        "actions": "操作",
        "process_image": "处理图像 (F5)",
        "save_processed": "保存处理后图像",
        "ready": "就绪",
        "keyboard_shortcuts": "键盘快捷键：",
        "open_image": "打开图像",
        "save_image": "保存图像",
        "process": "处理",
        "select_app_short": "选择应用",
        "full_size": "完整尺寸：",
        "tracking": "跟踪：",
        "no_app_selected": "未选择要跟踪的应用程序",
        "press_space": "按空格键捕获",
        "processing_time": "处理时间：{:.1f} 毫秒",
        "processing_complete": "处理完成",
        "upscaled": "已放大 {}x 使用 {}",
        "select_app_to_track": "选择要跟踪的应用程序：",
        "refresh": "刷新",
        "warning": "警告",
        "please_select": "请选择一个应用程序",
        "error": "错误",
        "language": "语言："
    },
    "zh_TW": {
        "app_title": "高級圖像處理器",
        "original_image": "原始圖像",
        "processed_image": "處理後圖像",
        "image_source": "圖像來源",
        "upload_image": "上傳圖像",
        "select_app": "選擇應用程序",
        "capturing_space": "捕獲：跟踪時按空格鍵",
        "enhancement_options": "增強選項",
        "mode": "模式：",
        "standard": "標準",
        "card_symbols": "卡片符號",
        "resolution_scale": "解析度縮放：",
        "method": "方法：",
        "downscale_speed": "縮小以提高速度",
        "display_options": "顯示選項",
        "preview_size": "預覽大小：",
        "small": "小",
        "medium": "中",
        "large": "大",
        "zoom": "縮放：",
        "actions": "操作",
        "process_image": "處理圖像 (F5)",
        "save_processed": "保存處理後圖像",
        "ready": "就緒",
        "keyboard_shortcuts": "鍵盤快捷鍵：",
        "open_image": "打開圖像",
        "save_image": "保存圖像",
        "process": "處理",
        "select_app_short": "選擇應用",
        "full_size": "完整尺寸：",
        "tracking": "跟踪：",
        "no_app_selected": "未選擇要跟踪的應用程序",
        "press_space": "按空格鍵捕獲",
        "processing_time": "處理時間：{:.1f} 毫秒",
        "processing_complete": "處理完成",
        "upscaled": "已放大 {}x 使用 {}",
        "select_app_to_track": "選擇要跟踪的應用程序：",
        "refresh": "刷新",
        "warning": "警告",
        "please_select": "請選擇一個應用程序",
        "error": "錯誤",
        "language": "語言："
    }
}

class AppSelector:
    def __init__(self, parent):
        self.parent = parent
        self.selected_window = None
        self.window_list = []
        
        # Get DPI scale factor from parent if it's our ModernImageProcessor
        self.dpi_scale = 1.0
        self.current_language = "zh_TW"  # Use Traditional Chinese
        if hasattr(parent, 'dpi_scale'):
            self.dpi_scale = parent.dpi_scale
        
        # Get translations
        self.translations = TRANSLATIONS["zh_TW"]
        
        # Scale font size based on DPI
        font_size = max(10, int(10 * self.dpi_scale) + 10)
        
        # Create a new toplevel window
        self.selector = tk.Toplevel(parent)
        self.selector.title(self.translations["select_app"])
        
        # Scale window size based on DPI
        width = int(1200 * self.dpi_scale)
        height = int(800 * self.dpi_scale)
        self.selector.geometry(f"{width}x{height}")
        self.selector.transient(parent)
        self.selector.grab_set()
        
        # Create widgets with scaled fonts
        ttk.Label(self.selector, text=self.translations["select_app_to_track"], 
                 font=("Segoe UI", font_size)).pack(pady=10)
        
        # Create frame for listbox and scrollbar
        frame = ttk.Frame(self.selector)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add listbox with scaled font
        self.app_listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, 
                                     font=("Segoe UI", font_size))
        self.app_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.app_listbox.yview)
        
        # Refresh and OK buttons
        btn_frame = ttk.Frame(self.selector)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Use scaled padding
        padx = max(5, int(5 * self.dpi_scale) + 10)
        
        ttk.Button(btn_frame, text=self.translations["refresh"], command=self.populate_app_list,
                  style="TButton").pack(side=tk.LEFT, padx=padx)
        ttk.Button(btn_frame, text="OK", command=self.on_select,
                  style="TButton").pack(side=tk.RIGHT, padx=padx)
        
        # Populate the list
        self.populate_app_list()
        
        # Wait for the window to be closed
        parent.wait_window(self.selector)
    
    def populate_app_list(self):
        self.app_listbox.delete(0, tk.END)
        self.window_list = []
        
        def enum_windows_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) != "":
                title = win32gui.GetWindowText(hwnd)
                if title and title != self.translations["select_app_to_track"]:
                    self.window_list.append((title, hwnd))
                    self.app_listbox.insert(tk.END, title)
        
        win32gui.EnumWindows(enum_windows_callback, None)
    
    def on_select(self):
        selected_idx = self.app_listbox.curselection()
        if selected_idx:
            self.selected_window = self.window_list[selected_idx[0]]
            self.selector.destroy()
        else:
            messagebox.showwarning(self.translations["warning"], self.translations["please_select"])
    
    def get_selected_window(self):
        return self.selected_window

class ModernImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("高級圖像處理器")  # Set Chinese title directly
        
        # Set DPI awareness at the beginning
        self.set_dpi_awareness()
        
        # Get DPI scale factor and set initial zoom factor to compensate 
        self.dpi_scale = self.get_dpi_scale_factor()
        initial_zoom = 2.0  # Use half the DPI scale as default zoom
        
        # Set language to Traditional Chinese
        self.current_language = "zh_TW"  # Use Traditional Chinese
        self.translations = TRANSLATIONS["zh_TW"]
        
        # Calculate window size based on screen resolution
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set window size to 80% of screen size
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        
        # Ensure minimum window size
        width = max(2400, width)
        height = max(1200, height)
        
        # Set window geometry (only set size, not position)
        self.root.geometry(f"{width}x{height}")
        
        # Apply modern style
        self.setup_styles()
        
        # Set window title with translated text
        self.root.title(self.translations["app_title"])
        
        # Center window on screen using a more reliable method
        # This needs to be done after setting geometry and before further UI initialization
        self.root.eval('tk::PlaceWindow . center')
        
        # Adjust sidebar width based on DPI
        self.sidebar_width = int(250 * self.dpi_scale + 400)
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.enhancement_mode = tk.StringVar(value="standard")
        self.tracking_enabled = False
        self.tracked_window = None
        self.track_window_info = None
        self.original_image_resolution = (0, 0)
        
        # Scale display size based on DPI
        self.display_size = int(500 * self.dpi_scale + 800)  # Scale default display size
        self.zoom_factor = initial_zoom  # Start with half DPI as default zoom
        self.upscale_factor = tk.IntVar(value=2)  # Default 2x upscaling
        self.upscale_method = tk.StringVar(value="edgepreserving")  # Default method
        self.resize_before_processing = tk.BooleanVar(value=True)  # Resize before to speed up
        
        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Setup the modern UI
        self.setup_ui()
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # Select app to track after UI is created
        self.root.after(100, self.select_app_to_track)
    
    def set_dpi_awareness(self):
        """Set DPI awareness to ensure proper scaling on high-DPI displays"""
        try:
            # Try the modern API first (Windows 10)
            windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        except Exception:
            try:
                # Fallback to older API (Windows 8/8.1)
                windll.user32.SetProcessDPIAware()
            except Exception:
                # Older Windows versions don't support DPI awareness
                pass

    def setup_styles(self):
        """Configure the modern style for the application"""
        self.style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            self.style.theme_use("clam")  # Try a more modern theme first
        except tk.TclError:
            try:
                self.style.theme_use("vista")  # Windows alternative
            except tk.TclError:
                pass  # Use default theme if none available
        
        # Scale font sizes based on DPI scale factor
        base_font_size = int(10 * self.dpi_scale + 10)
        small_font_size = max(9, int(9 * self.dpi_scale) + 10)
        header_font_size = int(12 * self.dpi_scale + 10)
        subheader_font_size = int(11 * self.dpi_scale + 10)
        
        # Configure common styles with scaled fonts
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabel", background="#f5f5f5", font=("Segoe UI", base_font_size))
        self.style.configure("TButton", font=("Segoe UI", base_font_size))
        self.style.configure("Sidebar.TFrame", background="#e0e0e0")
        self.style.configure("Header.TLabel", font=("Segoe UI", header_font_size, "bold"))
        self.style.configure("Subheader.TLabel", font=("Segoe UI", subheader_font_size, "bold"))
        self.style.configure("Status.TLabel", font=("Segoe UI", small_font_size))
        self.style.configure("Primary.TButton", font=("Segoe UI", base_font_size, "bold"))
        
        # Custom styles for sidebar sections
        self.style.configure("Section.TLabelframe", background="#e0e0e0")
        self.style.configure("Section.TLabelframe.Label", background="#e0e0e0", 
                            font=("Segoe UI", subheader_font_size, "bold"))
        
        # Custom style for image display frames - use larger font size
        display_font_size = int(12 * self.dpi_scale + 12)  # Larger font for image display frames
        self.style.configure("Display.TLabelframe", background="#f5f5f5")
        self.style.configure("Display.TLabelframe.Label", background="#f5f5f5", 
                            font=("Segoe UI", display_font_size, "bold"))
        
        # Progress bar style
        self.style.configure("TProgressbar", thickness=8)
        
        # Scrollbar style - make width proportional to DPI scale
        scrollbar_width = max(16, int(16 * self.dpi_scale))
        self.style.configure("TScrollbar", thickness=scrollbar_width, width=scrollbar_width, arrowsize=scrollbar_width)
        
        # Configure fonts for specific widgets that may not inherit properly
        self.root.option_add("*TCombobox*Listbox*Font", ("Segoe UI", base_font_size))
        self.root.option_add("*Font", ("Segoe UI", base_font_size))
        
        # Scale option menu font
        self.style.configure("TMenubutton", font=("Segoe UI", base_font_size))
        
        # Update default menu font (for dropdown menus)
        self.root.option_add("*Menu.font", ("Segoe UI", base_font_size))
        
        # Scale radio and check buttons
        self.style.configure("TRadiobutton", font=("Segoe UI", base_font_size))
        self.style.configure("TCheckbutton", font=("Segoe UI", base_font_size))
        
        # Print font size info to console for debugging
        print(f"DPI Scale: {self.dpi_scale:.2f}x - Font sizes: Base={base_font_size}, Header={header_font_size}, Small={small_font_size}")
        
    def get_dpi_scale_factor(self):
        """Get the DPI scale factor for the current monitor"""
        try:
            # Get DPI for the primary monitor
            awareness = windll.shcore.GetProcessDpiAwareness(0)
            if awareness >= 1:  # DPI aware
                hdc = windll.user32.GetDC(None)
                dpi_x = windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                windll.user32.ReleaseDC(None, hdc)
                return dpi_x / 96.0  # 96 is the default DPI
            return 1.0
        except Exception:
            return 1.0  # Default scale factor
    
    def select_app_to_track(self):
        app_selector = AppSelector(self.root)
        selected_window = app_selector.get_selected_window()
        
        if selected_window:
            self.tracked_window = selected_window
            window_title, window_handle = selected_window
            self.track_window_info = (window_title, window_handle)
            self.status_label.config(text=f"{self.translations['tracking']} {window_title}")
            
            # Enable keyboard tracking
            self.setup_keyboard_listener()
            self.tracking_enabled = True
        else:
            self.status_label.config(text=self.translations["no_app_selected"])
    
    def setup_keyboard_listener(self):
        # Set up keyboard hook for SPACE key
        keyboard.on_press_key("space", self.on_space_pressed)
        self.status_label.config(text=f"{self.translations['tracking']} {self.track_window_info[0]} - {self.translations['press_space']}")
    
    def on_space_pressed(self, e):
        if self.tracking_enabled and self.track_window_info:
            # Capture screenshot in a separate thread to avoid UI freezing
            threading.Thread(target=self.capture_app_screenshot).start()
    
    def get_window_rect(self, hwnd):
        """Get the window rectangle with appropriate adjustments for different window types"""
        # Try to get the client area first (which excludes window borders and title bar)
        try:
            # Get the client rect 
            client_rect = win32gui.GetClientRect(hwnd)
            # Convert client coordinates to screen coordinates
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
            right, bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
            return (left, top, right, bottom)
        except:
            # Fallback to window rect if client rect fails
            window_rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = window_rect
            
            # For Explorer windows, we might need to adjust to exclude the title bar
            window_class = win32gui.GetClassName(hwnd)
            if "Explorer" in window_class or "CabinetWClass" in window_class:
                # Adjust to skip title bar (approximate)
                top += 30  # Skip title bar
            
            return (left, top, right, bottom)
    
    def capture_window_with_printwindow(self, hwnd):
        """Capture window using PrintWindow which works with hardware-accelerated content"""
        try:
            # Get window dimensions
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid window dimensions: {width}x{height}")
            
            # Get window DC and create a compatible DC
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create a bitmap to save the screen contents
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Use PrintWindow to capture the window content (including GPU-rendered content)
            # Try with PW_RENDERFULLCONTENT first (for Chrome/modern apps)
            result = user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
            
            # If that fails or results in empty image, try regular PrintWindow
            if not result:
                result = user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
            
            # Convert to PIL Image
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1)
            
            # Clean up
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
            # Check if the image is just black (might mean capture failed)
            # Calculate the average brightness (simplified check)
            img_array = np.array(img)
            avg_brightness = np.mean(img_array)
            
            if avg_brightness < 5.0:  # Almost black
                self.root.after(0, lambda: self.status_label.config(
                    text="Warning: Captured image appears black. Trying fallback method..."
                ))
                raise ValueError("Captured image appears to be black")
            
            return img, (width, height)
        
        except Exception as e:
            # Log the exception and continue to fallback methods
            self.root.after(0, lambda: self.status_label.config(
                text=f"PrintWindow failed: {str(e)}. Trying fallback..."
            ))
            raise e
    
    def capture_window_content(self, hwnd):
        """Capture window content using multiple methods for reliability"""
        
        # Method priority:
        # 1. PrintWindow with PW_RENDERFULLCONTENT (for Chrome and hardware-accelerated apps)
        # 2. BitBlt from window DC (standard windows)
        # 3. ImageGrab (fallback)
        
        methods_to_try = [
            # Method 1: PrintWindow (best for Chrome/hardware acceleration)
            lambda: self.capture_window_with_printwindow(hwnd),
            
            # Method 2: BitBlt direct
            lambda: self.capture_with_bitblt(hwnd),
            
            # Method 3: ImageGrab
            lambda: self.capture_with_imagegrab(hwnd)
        ]
        
        last_exception = None
        for method in methods_to_try:
            try:
                result = method()
                img, dimensions = result
                
                # Verify result isn't completely black
                img_array = np.array(img)
                if np.mean(img_array) < 3.0:
                    continue  # Try next method if image is essentially black
                
                return result
            except Exception as e:
                last_exception = e
                continue
        
        # If all methods failed
        raise Exception(f"All capture methods failed. Last error: {str(last_exception)}")
    
    def capture_with_bitblt(self, hwnd):
        """Capture using traditional BitBlt method"""
        left, top, right, bottom = self.get_window_rect(hwnd)
        width, height = right - left, bottom - top
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid window dimensions: {width}x{height}")
        
        # Create a device context
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        
        # Create a bitmap to save the screen contents
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        
        # Copy the screen into the bitmap
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
        
        # Convert to PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)
        
        # Clean up
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        
        return img, (width, height)
    
    def capture_with_imagegrab(self, hwnd):
        """Capture using PIL's ImageGrab - fallback method"""
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width, height = right - left, bottom - top
        
        # Make sure the window is in the foreground before capturing
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)  # Give it time to come to the foreground
        
        # Capture using PIL's ImageGrab
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        return img, (width, height)
    
    def capture_app_screenshot(self):
        try:
            # Store current zoom setting to preserve it
            current_zoom = self.zoom_factor
            
            # Get window handle
            window_title, hwnd = self.track_window_info
            
            # Bring window to foreground
            try:
                # Check if the window still exists
                if not win32gui.IsWindow(hwnd):
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "The selected window is no longer available. Please select a new window."))
                    return
                
                # Make window visible
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.3)  # Wait for window to become active
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Warning: Could not focus window: {str(e)}"
                ))
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text=f"Capturing window: {window_title}..."
            ))
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(10))
            
            # Capture the window content
            screenshot, (width, height) = self.capture_window_content(hwnd)
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(30))
            
            # Update status with capture information
            self.root.after(0, lambda: self.status_label.config(
                text=f"Captured: {window_title} ({width}x{height}px)"
            ))
            
            # Convert to numpy array for OpenCV processing
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Optionally resize to smaller resolution for faster processing
            if self.resize_before_processing.get() and max(width, height) > 1000:
                scale = 1000 / max(width, height)
                new_width, new_height = int(width * scale), int(height * scale)
                screenshot_cv = cv2.resize(screenshot_cv, (new_width, new_height), 
                                          interpolation=cv2.INTER_AREA)
                width, height = new_width, new_height
            
            # Store original resolution
            self.original_image_resolution = (width, height)
            
            # Update the GUI with the screenshot
            self.original_image = screenshot_cv  # Keep full resolution for processing
            
            # Convert to RGB for display
            rgb_image = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2RGB)
            
            # Force the zoom factor to be exactly what it was before
            # Apply the exact same zoom as before capture
            self.zoom_var.set(current_zoom)  # Ensure zoom slider stays consistent
            rgb_image = self.apply_zoom(rgb_image, current_zoom)
            
            # For zoom factors > 1, don't constrain display size as much
            max_display_size = self.display_size
            if current_zoom > 1.0:
                max_display_size = int(self.display_size * (1.0 + (current_zoom - 1.0) * 0.5))
            
            # Don't resize if image is already smaller than max_display_size
            height, width = rgb_image.shape[:2]
            if height <= max_display_size and width <= max_display_size:
                display_image = rgb_image
            else:
                display_image = self.resize_image(rgb_image, max_display_size)
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(40))
            
            # Convert to PhotoImage and display in main thread
            self.root.after(0, self.update_original_display, display_image)
            
            # Process the screenshot
            self.process_image()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error capturing screenshot: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.status_label.config(text="Error capturing screenshot"))
    
    def apply_zoom(self, image, zoom_factor):
        """Apply zoom to show small elements better"""
        # Direct return if zoom is 1.0 (no change)
        if zoom_factor == 1.0:
            return image
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        # Apply interpolation based on zoom direction
        interpolation = cv2.INTER_AREA if zoom_factor < 1.0 else cv2.INTER_LINEAR
        
        # Resize the image with appropriate interpolation
        try:
            return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        except Exception as e:
            print(f"Zoom error: {e}")
            return image  # Return original if resize fails
    
    def update_original_display(self, display_image):
        """Update the display of the original image"""
        try:
            # Create PhotoImage from numpy array
            self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
            self.original_label.configure(image=self.original_photo)
            
            # Update resolution info with zoom info if needed
            zoom_info = f" (Zoomed {self.zoom_factor:.1f}x)" if self.zoom_factor != 1.0 else ""
            self.original_res_label.config(
                text=f"{self.translations['full_size']} {self.original_image_resolution[0]}x{self.original_image_resolution[1]}px{zoom_info}"
            )
            
            # Update scroll region for larger images that need scrolling
            self.original_canvas.update_idletasks()  # Force update to get correct bbox
            self.original_canvas.configure(scrollregion=self.original_canvas.bbox("all"))
            
            # Enable process button
            self.process_btn.state(['!disabled'])
        except Exception as e:
            messagebox.showerror(self.translations["error"], f"{self.translations['error']} updating display: {str(e)}")
    
    def update_zoom(self, *args):
        """Update the zoom factor based on slider value"""
        self.zoom_factor = self.zoom_var.get()
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
        
        # Update display if we have images
        if self.original_image is not None:
            # Apply zoom to original
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            zoomed_image = self.apply_zoom(rgb_image, self.zoom_factor)
            
            # Dynamically adjust display size based on zoom factor
            # Allow larger display size for higher zoom factors
            if self.zoom_factor <= 1.0:
                # For zoom factors <= 1, use standard display size
                max_display_size = self.display_size
            else:
                # For zoom factors > 1, use a more appropriate display size
                # that scales with zoom but doesn't get too large
                max_display_size = int(self.display_size * (1.0 + (self.zoom_factor - 1.0) * 0.5))
            
            # Don't resize if image is already smaller than max_display_size
            height, width = zoomed_image.shape[:2]
            if height <= max_display_size and width <= max_display_size:
                display_image = zoomed_image
            else:
                display_image = self.resize_image(zoomed_image, max_display_size)
            
            self.update_original_display(display_image)
        
        if self.processed_image is not None:
            # Apply zoom to processed
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            zoomed_image = self.apply_zoom(rgb_image, self.zoom_factor)
            
            # Dynamically adjust display size based on zoom factor
            # Allow larger display size for higher zoom factors
            if self.zoom_factor <= 1.0:
                # For zoom factors <= 1, use standard display size
                max_display_size = self.display_size
            else:
                # For zoom factors > 1, use a more appropriate display size
                # that scales with zoom but doesn't get too large
                max_display_size = int(self.display_size * (1.0 + (self.zoom_factor - 1.0) * 0.5))
            
            # Don't resize if image is already smaller than max_display_size
            height, width = zoomed_image.shape[:2]
            if height <= max_display_size and width <= max_display_size:
                display_image = zoomed_image
            else:
                display_image = self.resize_image(zoomed_image, max_display_size)
            
            self.processed_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
            self.processed_label.configure(image=self.processed_photo)
            
            # Update scroll region
            self.processed_canvas.update_idletasks()  # Force update to get correct bbox
            self.processed_canvas.configure(scrollregion=self.processed_canvas.bbox("all"))
            
            # Update resolution info with zoom
            zoom_info = f" (Zoomed {self.zoom_factor:.1f}x)" if self.zoom_factor != 1.0 else ""
            height, width = self.processed_image.shape[:2]
            self.processed_res_label.config(
                text=f"{self.translations['full_size']} {width}x{height}px{zoom_info}"
            )
    
    def set_zoom_preset(self, value):
        """Set zoom to a preset value"""
        self.zoom_var.set(value)
        self.update_zoom()
    
    def update_display_size(self):
        self.display_size = self.size_var.get()
        # If we have images loaded, update their display size
        if self.original_image is not None:
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            # Apply zoom first if needed
            if self.zoom_factor != 1.0:
                rgb_image = self.apply_zoom(rgb_image, self.zoom_factor)
            display_image = self.resize_image(rgb_image, self.display_size)
            self.update_original_display(display_image)
        
        if self.processed_image is not None:
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            # Apply zoom first if needed
            if self.zoom_factor != 1.0:
                rgb_image = self.apply_zoom(rgb_image, self.zoom_factor)
            display_image = self.resize_image(rgb_image, self.display_size)
            self.processed_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
            self.processed_label.configure(image=self.processed_photo)
            
            # Update scroll region
            self.processed_canvas.configure(scrollregion=self.processed_canvas.bbox("all"))
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                # Set the progress to show something is happening
                self.progress_var.set(10) 
                self.status_label.config(text=f"Loading image: {os.path.basename(file_path)}")
                
                # Read and display original image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", "Could not load image")
                    self.progress_var.set(0)
                    self.status_label.config(text="Error loading image")
                    return
                
                # Store original resolution
                height, width = self.original_image.shape[:2]
                self.original_image_resolution = (width, height)
                
                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Apply zoom if needed
                if self.zoom_factor != 1.0:
                    rgb_image = self.apply_zoom(rgb_image, self.zoom_factor)
                
                # Resize image for display while maintaining aspect ratio
                display_image = self.resize_image(rgb_image, self.display_size)
                
                # Set progress to indicate display updating
                self.progress_var.set(40)
                
                # Convert to PhotoImage and display
                self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
                self.original_label.configure(image=self.original_photo)
                
                # Update scrollregion
                self.original_canvas.configure(scrollregion=self.original_canvas.bbox("all"))
                
                # Update resolution info
                zoom_info = f" (Zoomed {self.zoom_factor:.1f}x)" if self.zoom_factor != 1.0 else ""
                self.original_res_label.config(
                    text=f"{self.translations['full_size']} {width}x{height}px{zoom_info}"
                )
                
                # Enable process button
                self.process_btn.state(['!disabled'])
                self.processed_label.configure(image='')
                
                # Reset time label
                self.time_label.configure(text="Ready")
                
                # Complete progress indication
                self.progress_var.set(100)
                self.status_label.config(
                    text=f"Loaded: {os.path.basename(file_path)} ({width}x{height}px)"
                )
                
                # Reset progress bar after a delay
                self.root.after(1000, lambda: self.progress_var.set(0))
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
                self.progress_var.set(0)
                self.status_label.config(text="Error loading image")
    
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
        self.process_btn.state(['disabled'])
        self.save_btn.state(['disabled'])
        self.progress_var.set(10)
        self.status_label.config(text="Processing image...")
        
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
            self.root.after(0, lambda: self.status_label.config(text="Image preprocessing..."))
            
            # Check if we're going to upscale the image
            upscale_factor = self.upscale_factor.get()
            upscale_method = self.upscale_method.get()
            do_upscale = upscale_factor > 1
            
            # Process based on selected enhancement mode
            if self.enhancement_mode.get() == "card_symbols":
                # Use specialized card symbol enhancement
                self.root.after(0, lambda: self.status_label.config(text="Enhancing card symbols..."))
                processed = enhance_card_symbols(gray)
                self.progress_var.set(70)
            else:
                # Use standard denoising pipeline
                self.root.after(0, lambda: self.status_label.config(text="Applying noise reduction..."))
                denoised = fast_denoise(gray)
                self.progress_var.set(40)
                
                # Apply sharpening
                self.root.after(0, lambda: self.status_label.config(text="Enhancing details..."))
                sharpened = fast_sharpen(denoised)
                self.progress_var.set(60)
                
                # Apply contrast enhancement
                self.root.after(0, lambda: self.status_label.config(text="Optimizing contrast..."))
                processed = fast_contrast_enhance(sharpened)
                self.progress_var.set(70)
            
            # Apply super-resolution if needed
            if do_upscale:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Upscaling to {upscale_factor}x resolution using {upscale_method} method..."
                ))
                
                # Apply resolution enhancement
                self.processed_image = enhance_resolution(processed, upscale_factor, upscale_method)
                self.progress_var.set(90)
            else:
                # No upscaling needed
                self.processed_image = processed
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Record processed image resolution
            proc_height, proc_width = self.processed_image.shape[:2]
            
            # Convert to RGB for display
            display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            
            # Apply zoom if needed
            if self.zoom_factor != 1.0:
                display_image = self.apply_zoom(display_image, self.zoom_factor)
            
            # For zoom factors > 1, don't constrain display size as much
            max_display_size = self.display_size
            if self.zoom_factor > 1.0:
                max_display_size = int(self.display_size * (1.0 + (self.zoom_factor - 1.0) * 0.5))
            
            # Don't resize if image is already smaller than max_display_size
            height, width = display_image.shape[:2]
            if height <= max_display_size and width <= max_display_size:
                # Just use the zoomed image directly
                pass
            else:
                display_image = self.resize_image(display_image, max_display_size)
            
            # Update GUI in main thread
            self.root.after(0, self.update_processed_display, display_image, processing_time, (proc_width, proc_height))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Error processing image"))
            self.root.after(0, lambda: self.progress_var.set(0))
        finally:
            self.root.after(0, self.processing_complete)
    
    def update_processed_display(self, display_image, processing_time, resolution=None):
        try:
            self.processed_photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
            self.processed_label.configure(image=self.processed_photo)
            self.time_label.configure(text=self.translations["processing_time"].format(processing_time))
            
            # Update scrollregion
            self.processed_canvas.update_idletasks()  # Force update to get correct bbox
            self.processed_canvas.configure(scrollregion=self.processed_canvas.bbox("all"))
            
            if resolution:
                zoom_info = f" (Zoomed {self.zoom_factor:.1f}x)" if self.zoom_factor != 1.0 else ""
                self.processed_res_label.config(
                    text=f"{self.translations['full_size']} {resolution[0]}x{resolution[1]}px{zoom_info}"
                )
        except Exception as e:
            messagebox.showerror(self.translations["error"], f"{self.translations['error']} updating processed display: {str(e)}")
    
    def processing_complete(self):
        self.progress_var.set(100)
        self.process_btn.state(['!disabled'])
        self.save_btn.state(['!disabled'])
        
        # Update status with the upscale information if used
        if self.upscale_factor.get() > 1:
            upscale_info = self.translations["upscaled"].format(self.upscale_factor.get(), self.upscale_method.get())
            if self.track_window_info:
                self.status_label.config(text=f"{self.translations['processing_complete']} ({upscale_info}) - {self.translations['press_space']}")
            else:
                self.status_label.config(text=f"{self.translations['processing_complete']} ({upscale_info})")
        else:
            if self.track_window_info:
                self.status_label.config(text=f"{self.translations['processing_complete']} - {self.translations['press_space']}")
            else:
                self.status_label.config(text=self.translations["processing_complete"])
        
        # Reset progress bar after a delay
        self.root.after(2000, lambda: self.progress_var.set(0))
    
    def save_image(self):
        if self.processed_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Update progress and status
                self.progress_var.set(20)
                self.status_label.config(text="Saving image...")
                
                # Option to save zoomed version if zoom is active
                if self.zoom_factor != 1.0:
                    save_zoomed = messagebox.askyesno("Save Options", 
                                                     "Do you want to save the zoomed version?\n\nYes - Save with zoom applied\nNo - Save original resolution")
                    
                    if save_zoomed:
                        # Apply zoom to processed image before saving
                        height, width = self.processed_image.shape[:2]
                        new_height, new_width = int(height * self.zoom_factor), int(width * self.zoom_factor)
                        zoomed_image = cv2.resize(self.processed_image, (new_width, new_height), 
                                                interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(file_path, zoomed_image)
                        
                        self.progress_var.set(100)
                        self.status_label.config(text=f"Image saved with zoom ({self.zoom_factor:.1f}x) at {new_width}x{new_height}px")
                        messagebox.showinfo("Success", 
                                           f"Image saved with zoom ({self.zoom_factor:.1f}x) at {new_width}x{new_height}px!")
                    else:
                        # Save original resolution
                        cv2.imwrite(file_path, self.processed_image)
                        height, width = self.processed_image.shape[:2]
                        
                        self.progress_var.set(100)
                        self.status_label.config(text=f"Image saved at original resolution ({width}x{height}px)")
                        messagebox.showinfo("Success", f"Image saved at original resolution ({width}x{height}px)!")
                else:
                    # Save at original resolution
                    cv2.imwrite(file_path, self.processed_image)
                    height, width = self.processed_image.shape[:2]
                    
                    self.progress_var.set(100)
                    self.status_label.config(text=f"Image saved at resolution ({width}x{height}px)")
                    messagebox.showinfo("Success", f"Image saved at full resolution ({width}x{height}px)!")
                
                # Reset progress bar after a delay
                self.root.after(2000, lambda: self.progress_var.set(0))
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")
                self.status_label.config(text="Error saving image")
                self.progress_var.set(0)
                
    def on_closing(self):
        # Clean up keyboard listener if enabled
        if self.tracking_enabled:
            keyboard.unhook_all()
        self.root.destroy()
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-p>", lambda e: self.process_image())
        self.root.bind("<Control-a>", lambda e: self.select_app_to_track())
        self.root.bind("<F5>", lambda e: self.process_image())
    
    def setup_ui(self):
        """Create the modern UI with sidebar and content panels"""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create sidebar frame with fixed width
        sidebar_frame = ttk.Frame(self.main_container, width=self.sidebar_width)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create a canvas inside the sidebar frame for scrolling
        sidebar_canvas = tk.Canvas(sidebar_frame, background="#e0e0e0", highlightthickness=0)
        sidebar_scrollbar = ttk.Scrollbar(sidebar_frame, orient=tk.VERTICAL, command=sidebar_canvas.yview)
        
        # Pack scrollbar and canvas
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        
        # Create frame inside canvas for sidebar content
        self.sidebar = ttk.Frame(sidebar_canvas, style="Sidebar.TFrame")
        sidebar_window = sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor=tk.NW, width=self.sidebar_width - 20)
        
        # Configure scroll region when sidebar size changes
        def configure_scroll_region(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
            # Also ensure the inner frame matches canvas width
            sidebar_canvas.itemconfig(sidebar_window, width=sidebar_canvas.winfo_width())
        
        self.sidebar.bind("<Configure>", configure_scroll_region)
        
        # Bind mouse wheel to sidebar scrolling
        def _on_mousewheel(event):
            sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        sidebar_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.sidebar.bind("<MouseWheel>", _on_mousewheel)
        
        # Content area (contains image displays)
        self.content_area = ttk.Frame(self.main_container)
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create sections in the sidebar
        self.create_sidebar_sections()
        
        # Create image display area
        self.create_image_display_area()
        
        # Create status bar
        self.create_status_bar()
        
        # No need to center again - already centered during initialization
    
    def create_sidebar_sections(self):
        """Create organized sections in the sidebar"""
        # Calculate scaled padding values
        padx = max(10, int(10 * self.dpi_scale))
        pady = max(5, int(5 * self.dpi_scale))
        pady_large = max(10, int(10 * self.dpi_scale))
        button_width = max(20, int(20 * self.dpi_scale))
        
        # Get parent scrollable canvas for mouse wheel binding
        sidebar_canvas = self.sidebar.master
        
        # Function to bind mousewheel to any widget
        def bind_mousewheel_to_widget(widget):
            widget.bind("<MouseWheel>", lambda e: sidebar_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
            for child in widget.winfo_children():
                bind_mousewheel_to_widget(child)
        
        # App title and info
        title_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        title_frame.pack(fill=tk.X, padx=padx, pady=(int(15 * self.dpi_scale), pady))
        
        ttk.Label(title_frame, text=self.translations["app_title"], 
                 style="Header.TLabel").pack(anchor=tk.W)
        
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=padx + 50, pady=pady_large)
        
        # Source section
        source_frame = ttk.LabelFrame(self.sidebar, text=self.translations["image_source"], style="Section.TLabelframe")
        source_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        ttk.Button(source_frame, text=self.translations["upload_image"], 
                  command=self.upload_image, width=button_width).pack(padx=padx, pady=pady)
        
        ttk.Button(source_frame, text=self.translations["select_app"], 
                  command=self.select_app_to_track, width=button_width).pack(padx=padx, pady=pady)
        
        ttk.Label(source_frame, text=self.translations["capturing_space"],
                 style="Status.TLabel").pack(padx=padx, pady=(0, pady), anchor=tk.W)
        
        # Enhancement options
        enhance_frame = ttk.LabelFrame(self.sidebar, text=self.translations["enhancement_options"], style="Section.TLabelframe")
        enhance_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        # Mode selection
        ttk.Label(enhance_frame, text=self.translations["mode"], style="Subheader.TLabel").pack(anchor=tk.W, padx=padx, pady=(pady, 0))
        
        mode_frame = ttk.Frame(enhance_frame)
        mode_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        ttk.Radiobutton(mode_frame, text=self.translations["standard"], variable=self.enhancement_mode, 
                       value="standard").pack(side=tk.LEFT, padx=(0, int(10 * self.dpi_scale)))
        ttk.Radiobutton(mode_frame, text=self.translations["card_symbols"], variable=self.enhancement_mode, 
                       value="card_symbols").pack(side=tk.LEFT)
        
        # Upscale options
        ttk.Label(enhance_frame, text=self.translations["resolution_scale"], style="Subheader.TLabel").pack(anchor=tk.W, padx=padx, pady=(pady, 0))
        
        scale_frame = ttk.Frame(enhance_frame)
        scale_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        rb_padx = int(10 * self.dpi_scale)
        ttk.Radiobutton(scale_frame, text="1x", variable=self.upscale_factor, 
                       value=1).pack(side=tk.LEFT, padx=(0, rb_padx))
        ttk.Radiobutton(scale_frame, text="2x", variable=self.upscale_factor, 
                       value=2).pack(side=tk.LEFT, padx=(0, rb_padx))
        ttk.Radiobutton(scale_frame, text="4x", variable=self.upscale_factor, 
                       value=4).pack(side=tk.LEFT)
        
        # Upscale method
        ttk.Label(enhance_frame, text=self.translations["method"], style="Subheader.TLabel").pack(anchor=tk.W, padx=padx, pady=(pady, 0))
        
        method_frame = ttk.Frame(enhance_frame)
        method_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        ttk.OptionMenu(method_frame, self.upscale_method, "edgepreserving", 
                      "simple", "edgepreserving", "detail").pack(fill=tk.X)
        
        # Speed option
        speed_frame = ttk.Frame(enhance_frame)
        speed_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        ttk.Checkbutton(speed_frame, text=self.translations["downscale_speed"], 
                       variable=self.resize_before_processing).pack(anchor=tk.W)
        
        # Display options
        display_frame = ttk.LabelFrame(self.sidebar, text=self.translations["display_options"], style="Section.TLabelframe")
        display_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        # Display size
        ttk.Label(display_frame, text=self.translations["preview_size"], style="Subheader.TLabel").pack(anchor=tk.W, padx=padx, pady=(pady, 0))
        
        size_frame = ttk.Frame(display_frame)
        size_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        # Calculate display size options scaled to DPI
        small_size = int(300 * self.dpi_scale)
        medium_size = int(500 * self.dpi_scale) 
        large_size = int(700 * self.dpi_scale)
        
        self.size_var = tk.IntVar(value=self.display_size)
        ttk.Radiobutton(size_frame, text=self.translations["small"], variable=self.size_var, 
                       value=small_size, command=self.update_display_size).pack(side=tk.LEFT, padx=(0, rb_padx))
        ttk.Radiobutton(size_frame, text=self.translations["medium"], variable=self.size_var, 
                       value=medium_size, command=self.update_display_size).pack(side=tk.LEFT, padx=(0, rb_padx))
        ttk.Radiobutton(size_frame, text=self.translations["large"], variable=self.size_var, 
                       value=large_size, command=self.update_display_size).pack(side=tk.LEFT)
        
        # Zoom control
        ttk.Label(display_frame, text=self.translations["zoom"], style="Subheader.TLabel").pack(anchor=tk.W, padx=padx, pady=(pady, 0))
        
        zoom_frame = ttk.Frame(display_frame)
        zoom_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        self.zoom_var = tk.DoubleVar(value=self.zoom_factor)
        zoom_slider = ttk.Scale(zoom_frame, from_=0.2, to=5.0, variable=self.zoom_var, 
                               orient=tk.HORIZONTAL, length=int(220 * self.dpi_scale), command=self.update_zoom)
        zoom_slider.pack(fill=tk.X, pady=(0, pady))
        
        # Zoom presets and label in one row
        zoom_preset_frame = ttk.Frame(zoom_frame)
        zoom_preset_frame.pack(fill=tk.X)
        
        # Scale button width
        button_small_width = max(4, int(4 * self.dpi_scale))
        
        ttk.Button(zoom_preset_frame, text="0.5x", width=button_small_width, 
                  command=lambda: self.set_zoom_preset(0.5)).pack(side=tk.LEFT, padx=(0, pady))
        ttk.Button(zoom_preset_frame, text="1x", width=button_small_width, 
                  command=lambda: self.set_zoom_preset(1.0)).pack(side=tk.LEFT, padx=pady)
        ttk.Button(zoom_preset_frame, text="2x", width=button_small_width, 
                  command=lambda: self.set_zoom_preset(2.0)).pack(side=tk.LEFT, padx=pady)
        
        self.zoom_label = ttk.Label(zoom_preset_frame, text=f"{self.zoom_factor:.1f}x")
        self.zoom_label.pack(side=tk.RIGHT, padx=(pady, 0))
        
        # Action buttons
        action_frame = ttk.LabelFrame(self.sidebar, text=self.translations["actions"], style="Section.TLabelframe")
        action_frame.pack(fill=tk.X, padx=padx, pady=pady)
        
        self.process_btn = ttk.Button(action_frame, text=self.translations["process_image"], 
                                     command=self.process_image, style="Primary.TButton")
        self.process_btn.pack(fill=tk.X, padx=padx, pady=pady)
        self.process_btn.state(['disabled'])
        
        self.save_btn = ttk.Button(action_frame, text=self.translations["save_processed"], 
                                  command=self.save_image)
        self.save_btn.pack(fill=tk.X, padx=padx, pady=pady)
        self.save_btn.state(['disabled'])
        
        # Progress section
        progress_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        progress_frame.pack(fill=tk.X, padx=padx, pady=(pady, pady_large))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=0, pady=pady)
        
        self.time_label = ttk.Label(progress_frame, text=self.translations["ready"], style="Status.TLabel")
        self.time_label.pack(anchor=tk.W)
        
        # Add keyboard shortcut hints at the bottom
        hint_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        hint_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=padx, pady=pady_large)
        
        ttk.Label(hint_frame, text=self.translations["keyboard_shortcuts"], 
                 style="Subheader.TLabel").pack(anchor=tk.W, pady=(0, pady))
        
        shortcuts = [
            (self.translations["open_image"], "Ctrl+O"),
            (self.translations["save_image"], "Ctrl+S"),
            (self.translations["process"], "Ctrl+P or F5"),
            (self.translations["select_app_short"], "Ctrl+A")
        ]
        
        for action, key in shortcuts:
            shortcut_frame = ttk.Frame(hint_frame)
            shortcut_frame.pack(fill=tk.X, pady=2)
            ttk.Label(shortcut_frame, text=action, style="Status.TLabel").pack(side=tk.LEFT)
            ttk.Label(shortcut_frame, text=key, style="Status.TLabel").pack(side=tk.RIGHT)
        
        # Bind mousewheel event to all sidebar widgets for proper scrolling
        bind_mousewheel_to_widget(self.sidebar)
    
    def create_image_display_area(self):
        """Create the image display area with side-by-side panels"""
        # Image display container
        image_container = ttk.Frame(self.content_area)
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Create image display frames side by side
        self.original_frame = ttk.LabelFrame(image_container, text=self.translations["original_image"], style="Display.TLabelframe")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.processed_frame = ttk.LabelFrame(image_container, text=self.translations["processed_image"], style="Display.TLabelframe")
        self.processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create resolution info labels
        self.original_res_label = ttk.Label(self.original_frame, text=f"{self.translations['full_size']} None")
        self.original_res_label.pack(anchor=tk.NW, padx=5, pady=5)
        
        self.processed_res_label = ttk.Label(self.processed_frame, text=f"{self.translations['full_size']} None")
        self.processed_res_label.pack(anchor=tk.NW, padx=5, pady=5)
        
        # Create image labels with scrolling capability
        # Original image container with scrollbars
        self.original_container = ttk.Frame(self.original_frame)
        self.original_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_canvas = tk.Canvas(self.original_container, background="#f0f0f0")
        self.original_scrollbar_y = ttk.Scrollbar(self.original_container, orient=tk.VERTICAL, command=self.original_canvas.yview)
        self.original_scrollbar_x = ttk.Scrollbar(self.original_container, orient=tk.HORIZONTAL, command=self.original_canvas.xview)
        
        self.original_canvas.configure(yscrollcommand=self.original_scrollbar_y.set, xscrollcommand=self.original_scrollbar_x.set)
        
        self.original_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.original_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.original_frame_inner = ttk.Frame(self.original_canvas)
        self.original_canvas.create_window((0, 0), window=self.original_frame_inner, anchor=tk.NW)
        
        self.original_label = ttk.Label(self.original_frame_inner)
        self.original_label.pack()
        
        # Set canvas scrolling for mouse wheel
        self.original_canvas.bind("<MouseWheel>", lambda e: self.original_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.original_canvas.bind("<Control-MouseWheel>", lambda e: self.original_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Processed image container with scrollbars
        self.processed_container = ttk.Frame(self.processed_frame)
        self.processed_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(self.processed_container, background="#f0f0f0")
        self.processed_scrollbar_y = ttk.Scrollbar(self.processed_container, orient=tk.VERTICAL, command=self.processed_canvas.yview)
        self.processed_scrollbar_x = ttk.Scrollbar(self.processed_container, orient=tk.HORIZONTAL, command=self.processed_canvas.xview)
        
        self.processed_canvas.configure(yscrollcommand=self.processed_scrollbar_y.set, xscrollcommand=self.processed_scrollbar_x.set)
        
        self.processed_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.processed_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.processed_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.processed_frame_inner = ttk.Frame(self.processed_canvas)
        self.processed_canvas.create_window((0, 0), window=self.processed_frame_inner, anchor=tk.NW)
        
        self.processed_label = ttk.Label(self.processed_frame_inner)
        self.processed_label.pack()
        
        # Set canvas scrolling for mouse wheel
        self.processed_canvas.bind("<MouseWheel>", lambda e: self.processed_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.processed_canvas.bind("<Control-MouseWheel>", lambda e: self.processed_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Configure canvas resize events
        self.original_frame_inner.bind("<Configure>", 
                                     lambda e: self.original_canvas.configure(scrollregion=self.original_canvas.bbox("all")))
        self.processed_frame_inner.bind("<Configure>", 
                                      lambda e: self.processed_canvas.configure(scrollregion=self.processed_canvas.bbox("all")))
    
    def create_status_bar(self):
        """Create a status bar at the bottom of the window"""
        # Status bar at the bottom
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, style="Sidebar.TFrame")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status information
        self.status_label = ttk.Label(self.status_bar, text=self.translations["ready"], style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # DPI information on the right
        dpi_info = f"DPI Scale: {self.dpi_scale:.2f}x"
        self.dpi_label = ttk.Label(self.status_bar, text=dpi_info, style="Status.TLabel")
        self.dpi_label.pack(side=tk.RIGHT, padx=10, pady=5)

def main():
    root = tk.Tk()
    app = ModernImageProcessor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
