from PIL import Image
import os

def get_white_margins(img):
    """Calculates the width of the white margin on the left and right sides of an image."""
    # Convert to RGB if necessary to ensure consistent pixel values
    img = img.convert("RGB")
    width, height = img.size
    pixels = img.load()

    # Calculate left margin
    left_margin = 0
    for x in range(width):
        is_white_column = True
        for y in range(height):
            # Check if pixel is pure white
            if pixels[x, y] != (255, 255, 255):
                is_white_column = False
                break
        if is_white_column:
            left_margin += 1
        else:
            # Found the first non-white column, stop counting
            break

    # Calculate right margin
    right_margin = 0
    for x in range(width - 1, -1, -1):
        is_white_column = True
        for y in range(height):
            if pixels[x, y] != (255, 255, 255):
                is_white_column = False
                break
        if is_white_column:
            right_margin += 1
        else:
            break
            
    return left_margin, right_margin

# --- Main Script ---

# List of input image filenames
image_files = [f"redshift_turn_{i}.png" for i in range(1, 5)]

images = []
margins = []

print("Analyzing images to find margins...")
# 1. Load images and calculate their existing margins
for img_file in image_files:
    try:
        img = Image.open(img_file)
        images.append(img)
        lm, rm = get_white_margins(img)
        margins.append((lm, rm))
        print(f"  {img_file}: Left margin = {lm}, Right margin = {rm}")
    except FileNotFoundError:
        print(f"Error: Could not find {img_file}. Please make sure it's in the same directory.")
        exit()

# 2. Find the minimum left margin across all images
min_left_margin = min(m[0] for m in margins)
print(f"\nTarget margin size (smallest detected left margin): {min_left_margin} pixels")

# Create an output directory for the processed images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

print("\nProcessing and saving images...")
# 3. Crop each image and save the result
for i, (img, (current_left, current_right)) in enumerate(zip(images, margins)):
    width, height = img.size
    
    # The crop rectangle is defined by (left, top, right, bottom) coordinates.
    # We want to keep 'min_left_margin' pixels on both sides.
    
    # Crop from the left: Start at the current margin, move back by the target margin
    crop_left = current_left - min_left_margin
    
    # Crop from the right: End at the image width, move in by the current right margin,
    # then move out by the target margin.
    crop_right = width - (current_right - min_left_margin)
    
    # We are not cropping vertically
    crop_top = 0
    crop_bottom = height
    
    # Perform the crop
    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # Save the processed image
    output_filename = os.path.join(output_dir, image_files[i])
    cropped_img.save(output_filename)
    print(f"  Saved {output_filename}")

print(f"\nDone! Processed images can be found in the '{output_dir}' directory.")