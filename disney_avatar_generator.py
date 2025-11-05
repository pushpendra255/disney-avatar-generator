# ------------------------
# IMPORTS
# ------------------------
import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import torch
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from deepface import DeepFace

# ------------------------
# CONFIGURATION
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "stabilityai/stable-diffusion-2-1"

# Load Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)
pipe.enable_attention_slicing()

# ------------------------
# VALIDATION FUNCTIONS
# ------------------------
def validate_image(img_path):
    """Validate image format, resolution, and detect face coordinates."""
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError("Image must be JPEG or PNG")

    img = Image.open(img_path).convert("RGB")

    if img.width < 150 or img.height < 150:
        raise ValueError("Image resolution too low. Minimum 150x150 required.")

    # Face detection using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise RuntimeError("No face detected in the image")

    x, y, w, h = faces[0]
    return img, (x, y, w, h)

# ------------------------
# AGE & GENDER DETECTION
# ------------------------
def detect_age_gender(img_path):
    """Detect age and gender using DeepFace"""
    obj = DeepFace.analyze(img_path, actions=['age', 'gender'], enforce_detection=True)
    if isinstance(obj, list):
        obj = obj[0]
    return obj['age'], obj['gender']

# ------------------------
# BACKGROUND REMOVAL
# ------------------------
def remove_background(img_pil):
    """Remove background to make image transparent"""
    img_array = np.array(img_pil)
    result = remove(img_array)
    return Image.fromarray(result)

# ------------------------
# DISNEY AVATAR GENERATION
# ------------------------
def generate_disney_avatar(init_img, prompt=None):
    """Generate stylized Disney-like avatar using Stable Diffusion"""
    init_img = init_img.convert("RGB")
    width, height = init_img.size
    width, height = (width // 8) * 8, (height // 8) * 8
    init_img = init_img.resize((width, height))

    if prompt is None:
        prompt = (
            "3D Disney Pixar style portrait of the same person, ultra-detailed, "
            "cinematic lighting, colorful, soft background, digital art, smiling expression, "
            "no deformation, perfect symmetry, realistic eyes, smooth skin texture"
        )

    avatar = pipe(
        prompt=prompt,
        image=init_img,
        strength=0.7,
        guidance_scale=8.5,
        num_inference_steps=30
    ).images[0]

    return avatar

# ------------------------
# FULL PIPELINE
# ------------------------
if __name__ == "__main__":
    input_face = r"C:\Users\Pushpendra Singh\Desktop\Avatar\input_face.png"
    output_path = r"C:\Users\Pushpendra Singh\Desktop\Avatar\disney_avatar.png"

    # Step 1: Validate input
    img, face_coords = validate_image(input_face)
    print("✅ Image validated successfully!")

    # Step 2: Age & gender detection
    age, gender = detect_age_gender(input_face)
    print(f"🧠 Detected Age: {age}, Gender: {gender}")

    # Step 3: Remove background
    img_no_bg = remove_background(img)

    # Step 4: Generate Disney avatar
    avatar_img = generate_disney_avatar(img_no_bg)

    # Step 5: Ensure transparent background (RGBA)
    if avatar_img.mode != "RGBA":
        avatar_img = avatar_img.convert("RGBA")

    # Step 6: Save final output
    avatar_img.save(output_path)
    print("✅ Disney Avatar generated successfully!")
    print("📁 Saved at:", output_path)
