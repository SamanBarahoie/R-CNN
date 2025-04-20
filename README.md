

# 🖼️ Instance Segmentation with Mask R-CNN

This project demonstrates instance segmentation using a pre-trained **Mask R-CNN** model with a **ResNet-50 FPN** backbone from PyTorch's `torchvision` library. It processes an input image and visualizes the results with **segmentation masks**, **bounding boxes**, and **class labels**.

---

## 🚀 Features

- ✅ Loads a pre-trained Mask R-CNN model trained on the **COCO** dataset.
- 🧠 Performs **instance segmentation** on input images.
- 🎨 Visualizes segmentation results with customizable **confidence thresholds**.
- ⚙️ Supports both **CPU** and **GPU** execution.
- 🧩 Modular and well-documented codebase for easy extension.

---

## 📦 Requirements

- Python 3.7 or higher  
- PyTorch  
- torchvision  
- Pillow  
- Matplotlib  
- NumPy  

---

## 🔧 Installation

### 1. Clone the Repository (if applicable)
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
pip install torch torchvision pillow matplotlib numpy
```

Or with a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True if GPU is accessible
```

---

## ▶️ Usage

### 1. Prepare an Input Image
Place an image (e.g., `sample.jpg`) in the project directory or provide a path to it.

### 2. Run the Script
Edit the `main.py` file to set your image path:
```python
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    main(image_path)
```

Then run:
```bash
python main.py
```

### 3. View the Results
The script will generate a plot showing:
- 🟣 **Segmentation Masks**
- 🔴 **Bounding Boxes**
- 🏷️ **Class Labels & Confidence Scores**

Only predictions with confidence > 0.5 (by default) are shown.

---

## 📁 File Structure

```
├── main.py         # Main script
├── README.md       # Project documentation
└── sample.jpg      # Example image (optional)
```

---

A visualization displaying:
- ✅ Colored masks per object
- ✅ Red bounding boxes
- ✅ Class labels with scores

---

## 📝 Notes

- The Mask R-CNN model is trained on the **COCO dataset** (80 classes including `person`, `car`, `dog`, etc.).
- You can adjust `score_threshold` in the `visualize_results()` function to fine-tune filtering.
- For video or batch processing support, the code can be extended.

---

## ⚠️ Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed.
- **CUDA Errors**: If GPU is not available, the model will fall back to CPU automatically.
- **Slow Inference**: GPU is recommended for faster performance.

---

##  Need Help?

For feature requests or questions, feel free to open an issue or reach out to the maintainer.

---
