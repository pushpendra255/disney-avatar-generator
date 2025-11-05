# 🧚 Disney Avatar Generator

## 🎯 Objective
Automatically generate a 3D Disney/Pixar-style avatar from any human photo using **Stable Diffusion 2.1**.  
The system validates input, detects age & gender, removes background, and produces a clean, high-quality stylized avatar.

---

## 🚀 Tech Stack & Models Used
- **Stable Diffusion 2.1** → For style transformation (3D Disney look)
- **DeepFace** → Age & gender detection
- **rembg** → Background removal (transparent PNG)
- **OpenCV** → Face validation & detection
- **PyTorch** → For GPU-based inference
- **Diffusers Library (Hugging Face)** → Stable Diffusion Img2Img pipeline

---

## 🧠 Workflow
1. **Input Validation**
   - Ensures JPEG/PNG format
   - Checks minimum resolution (150×150)
   - Detects face using HaarCascade

2. **Age & Gender Detection**
   - Uses DeepFace to extract metadata before stylization.

3. **Background Removal**
   - Removes unwanted background using rembg (transparent output).

4. **Avatar Generation**
   - Applies Stable Diffusion 2.1 to create Pixar-style 3D avatar.

5. **Output**
   - Transparent PNG of stylized avatar with no deformation or quality loss.

---

## 🧩 Example Usage

```bash
python disney_avatar_generator.py
