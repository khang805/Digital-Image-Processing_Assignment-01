#  Digital Image Processing – Programming Assignment 1

## Overview

This assignment focuses on **fundamental image processing techniques** implemented **manually** using low-level operations with **NumPy** and **OpenCV**, without relying on high-level built-ins such as `cv2.equalizeHist()` or `np.histogram()`.

All functions are implemented from scratch inside **`dip_toolbox.py`**, and results are visualized using **Matplotlib**.

---

## 📂 Folder Structure

```
📁 DIP_Assignment1/
│
├── 22i-1148_AbdurrahmanKhan_dip_toolbox.py           # All implementations (Q1–Q5)
├── report.pdf               # Report with images, results & explanations
├── dip_assignment_images/   # Dataset folder
│   ├── cameraman.png
│   ├── horse.png
│   ├── coins.png
│   ├── astronaut.png
│   ├── chest_xray.png
│   ├── coffee.png
│   ├── satellite.png
│   ├── text.png
│
└── README.md                # This file
```

## 🧩 Tasks Summary

### **Q1: Adjacency & Connectivity**

**Images:** `cameraman.png`, `horse.png`

* Implemented **4-adjacency**, **8-adjacency**, and **m-adjacency** using **manual neighbor checks**.
* **Connectivity verification** between pixel coordinates using **BFS traversal**.
* **Manual queue** structure was used instead of Python’s `deque`.

**Concepts Used:**

* Binary thresholding
* Pixel adjacency logic
* BFS for connected component verification

**Observation:**
4-adjacency is more restrictive; 8-adjacency allows diagonal connectivity (possibly false positives); m-adjacency balances both.

---

### **Q2: Point-Based Image Enhancement**

**Images:**

* `coins.png` (for log & contrast stretching)
* `astronaut.png` (for gamma transformation)

**Techniques Implemented:**

1. **Logarithmic Transformation**

2. **Power-Law (Gamma) Transformation**

3. **Contrast Stretching (Piecewise Linear)**

**Observation:**
Log transform enhances dark regions, gamma < 1 brightens dark areas, and contrast stretching improves overall dynamic range.

---

### **Q3: Histogram Processing**

**Images:** `chest_xray.png`, `coffee.png`

**Implemented Methods:**

* Manual **Histogram**, **PDF**, **CDF** computation
* **Global Histogram Equalization**
* **Local Histogram Equalization** (window-based)
* **Histogram Specification (Matching)** between X-ray and Coffee images

**Observation:**
Histogram equalization improves contrast, while specification aligns tone distribution between different images.

---

### **Q4: Gray-Level Slicing**

**Image:** `satellite.png`

**Techniques Implemented:**

1. **With background preserved**
2. **With background suppressed**

**Observation:**
Highlights specific intensity ranges (e.g., 100–180) useful for extracting land/object regions from satellite images.

---

### **Q5: Thresholding**

**Images:** `coins.png`, `text.png`

**Implemented Methods:**

* **Global Thresholding**
* **Adaptive Thresholding (Mean & Median-based)**

**Observation:**
Adaptive methods outperform global thresholding for images with **non-uniform illumination** (e.g., text images).

---

##  Report Summary

The accompanying **`report.pdf`** contains:

* Important Mathematical Formulas
* Original vs. processed images
* Description of all questions

---

