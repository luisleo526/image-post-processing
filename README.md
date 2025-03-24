# Advanced Image Processor

A powerful image processing application with specialized algorithms for text and card symbol enhancement.

## Features

- **Modern UI**: Intuitive sidebar layout with organized controls
- **Capture Modes**: Upload images or capture from another application window
- **Enhancement Options**: Standard denoising or specialized card symbol enhancement
- **Resolution Control**: Upscale images up to 4x resolution with different algorithms
- **Zoom Control**: Easily view and save images with different zoom levels
- **Keyboard Shortcuts**:
  - `Ctrl+O`: Open image
  - `Ctrl+S`: Save processed image
  - `Ctrl+P` or `F5`: Process image
  - `Ctrl+A`: Select application to track
  - `Space`: Capture screenshot (when tracking an application)

## Usage

1. Run the application with `python main.py`
2. Select an image source:
   - Upload an image using the button or Ctrl+O
   - Select an application to track and capture screenshots with Space
3. Choose enhancement options from the sidebar
4. Click "Process Image" or press F5
5. Save the processed image with Ctrl+S

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```

## Build Cython Module

To build the optimized image processing module:
```
python setup.py build_ext --inplace
```


## Pyinstaller

```
pyinstaller --name "ImageProcessor" --onefile --windowed --add-data "image_processor.cp311-win_amd64.pyd;." main.py
```