from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    os.makedirs('static', exist_ok=True)
    size = 192
    img = Image.new('RGB', (size, size), color='#0d1117')
    draw = ImageDraw.Draw(img)
    
    # Draw border
    border_color = '#1D9E75'
    draw.rounded_rectangle([10, 10, size-10, size-10], radius=20, outline=border_color, width=8)
    
    # Draw text
    # Try to load a generic font or just use default if not available, but default is too small
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except IOError:
        font = ImageFont.load_default()
    
    text = "FI"
    # Get bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) / 2
    y = (size - text_height) / 2 - 10 # small offset for visual center
    
    draw.text((x, y), text, font=font, fill='white')
    img.save('static/icon.png')
    print("Icon generated at static/icon.png")

if __name__ == '__main__':
    create_icon()
