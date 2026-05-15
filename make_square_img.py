from PIL import Image

def make_square(input_path, output_path, fill_color=(0, 0, 0)):
    # Open image
    img = Image.open(input_path).convert("RGB")

    width, height = img.size
    new_size = min(width, height)

    # Create square canvas
    square_img = Image.new("RGB", (new_size, new_size), fill_color)

    # Paste original image centered
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2

    square_img.paste(img, (paste_x, paste_y))

    # Save as JPEG
    square_img.save(output_path, "JPEG")

    print(f"Saved square image to {output_path}")

# Example usage
make_square("lena.png", "s_lena.jpeg")
