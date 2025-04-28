import os
from PIL import Image
from pathlib import Path

def convert_png_to_laplace(input_dir: str = "data/lineImages"):
    """
    Convert all .png files in the specified directory and its subdirectories to laplace transform format.
    
    Args:
        input_dir (str): Path to the directory containing .png files
    """
    # Convert to Path object for better path handling
    input_path = Path(input_dir)
    
    # Check if directory exists
    if not input_path.exists():
        print(f"Directory {input_dir} does not exist!")
        return
    
    # Get all .tif files recursively
    png_files = list(input_path.rglob("*.png"))
    
    if not png_files:
        print(f"No .png files found in {input_dir} or its subdirectories")
        return
    
    # Convert each file
    for png_file in png_files:
        try:
            # Open the image
            with Image.open(png_file) as img:
                # Create output filename by replacing .png with .laplace
                output_file = png_file.with_suffix('.laplace')
                
                # Save as PNG
                img.save(output_file, 'PNG')
                print(f"Converted {png_file} to {output_file}")
                
        except Exception as e:
            print(f"Error converting {png_file}: {str(e)}")

if __name__ == "__main__":
    convert_png_to_laplace()
