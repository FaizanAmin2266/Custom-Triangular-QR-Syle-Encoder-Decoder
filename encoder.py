import numpy as np
from PIL import Image

def simple_custom_code(data_string, size=25):
    """
    data_string: the data to encode (ASCII text).
    size: the total width/height in cells. 
          (You might want to reserve some cells for orientation patterns.)
    """
    # 1 Convert data_string to binary representation
    #    For simplicity, assume ASCII, 8 bits per character.
    data_bits = []
    for char in data_string:
        bits = format(ord(char), '08b')
        data_bits.extend(int(b) for b in bits)
    
    # 2 Ensure we have enough cells (size*size) to hold these bits
    total_cells = size * size
    if len(data_bits) > total_cells:
        raise ValueError("Data is too long for chosen size.")
    
    # 3 Create a blank (white) grid
    code_array = np.zeros((size, size), dtype=np.uint8)
    
    # 4 Fill in bits in row-major order
    idx = 0
    for r in range(size):
        for c in range(size):
            if idx < len(data_bits):
                code_array[r, c] = data_bits[idx]
                idx += 1
            else:
                # If no more data bits, leave it at 0
                pass
    
    # 5 Convert 0/1 array to an actual image (white=255, black=0)
    #    Invert bits so that 1=black, 0=white
    code_image = (1 - code_array) * 255
    
    # 6 Convert to PIL image
    img = Image.fromarray(code_image.astype('uint8'), 'L')
    return img

def simple_custom_code_triangle(data_string, size=25):
    data_bits = []
    for char in data_string:
        data_bits.extend(int(b) for b in format(ord(char), '08b'))

    total_cells = (size * (size + 1)) // 2
    if len(data_bits) > total_cells:
        raise ValueError("Data too long for specified triangle.")

    import numpy as np
    code_array = np.zeros((size, size), dtype=np.uint8)

    idx = 0
    for r in range(size):
        # Calculate how many cells go in this row
        row_cells = r + 1
        # Offset from the left to center
        offset = (size - row_cells) // 2
        # Fill the bits in horizontally, starting at offset
        for c in range(row_cells):
            if idx < len(data_bits):
                code_array[r, offset + c] = data_bits[idx]
                idx += 1

    # Invert bits for black=1, white=0
    code_image = (1 - code_array) * 255
    from PIL import Image
    return Image.fromarray(code_image.astype('uint8'), 'L')

def simple_custom_code_triangle_anchor(data_string, size=25, buffer=2):
    import numpy as np
    from PIL import Image

    data_bits = []
    for char in data_string:
        data_bits.extend(int(b) for b in format(ord(char), '08b'))

    total_cells = (size * (size + 1)) // 2
    if len(data_bits) > total_cells:
        raise ValueError("Data too long for specified triangle.")

    # Create a blank array with extra buffer around
    output_size = size + buffer * 2
    code_array = np.zeros((output_size, output_size), dtype=np.uint8)

    anchor_size = 5
    center = output_size // 2  # center now shifted due to buffer

    # Draw the anchor at the top
    for r in range(anchor_size):
        for c in range(anchor_size):
            col_index = center - 2 + c
            # Draw a black border in anchor
            if r in [0, anchor_size - 1] or c in [0, anchor_size - 1]:
                code_array[r + buffer, col_index] = 1
    # Small black dot in the center
    code_array[buffer + 2, center] = 1

    # Fill data bits below the anchor
    idx = 0
    for r in range(anchor_size, size):
        row_cells = r + 1
        offset = (size - row_cells) // 2
        row_pos = r + buffer  # shift row by buffer
        col_offset = buffer + offset  # shift columns by buffer
        for c in range(row_cells):
            if idx < len(data_bits):
                code_array[row_pos, col_offset + c] = data_bits[idx]
                idx += 1

    # Convert to an actual image (1=black => pixel 0; 0=white => pixel 255)
    code_image = (1 - code_array) * 255
    return Image.fromarray(code_image.astype('uint8'), 'L')

def tsimple_custom_code_triangle_anchor(data_string, buffer=2):
    import numpy as np
    from PIL import Image

    # 1) Convert data to bits
    data_bits = []
    for char in data_string:
        data_bits.extend(int(b) for b in format(ord(char), '08b'))

    # 2) Find the smallest triangle size so (size*(size+1))/2 >= len(data_bits)
    needed = len(data_bits)
    tri_size = 1
    while (tri_size * (tri_size + 1)) // 2 < needed:
        tri_size += 1

    # 3) Define anchor size (in rows/cols) and total output size:
    anchor_size = 5
    # We have: top anchor (anchor_size rows), data rows (tri_size),
    # bottom anchors (anchor_size rows) plus left/right buffer.
    output_size = tri_size + 2 * anchor_size + 2 * buffer

    # Create a blank array
    code_array = np.zeros((output_size, output_size), dtype=np.uint8)

    # We'll keep a horizontal center for the top anchor and data.
    center = output_size // 2

    # ----------------------------------------------------------------
    # TOP ANCHOR: centered horizontally at rows buffer..(buffer+anchor_size-1)
    for r in range(anchor_size):
        for c in range(anchor_size):
            col_index = center - (anchor_size // 2) + c
            # Draw black border
            if r in [0, anchor_size - 1] or c in [0, anchor_size - 1]:
                code_array[buffer + r, col_index] = 1
    # A small black dot in the center of the top anchor
    code_array[buffer + (anchor_size // 2), center] = 1

    # ----------------------------------------------------------------
    # DATA ROWS: dynamically centered
    idx = 0
    for r in range(tri_size):
        row_cells = r + 1
        row_pos = buffer + anchor_size + r
        # Center this row horizontally so row_cells is centered around 'center'
        col_offset = center - (row_cells // 2)
        for c in range(row_cells):
            if idx < len(data_bits):
                code_array[row_pos, col_offset + c] = data_bits[idx]
                idx += 1

    # ----------------------------------------------------------------
    # BOTTOM-LEFT ANCHOR: placed immediately after the last data row, left-aligned
    bottom_anchor_row = buffer + anchor_size + tri_size
    for r in range(anchor_size):
        for c in range(anchor_size):
            # Draw black border for the bottom-left anchor
            if r in [0, anchor_size - 1] or c in [0, anchor_size - 1]:
                code_array[bottom_anchor_row + r, buffer + c] = 1
    # Small black dot in the center of the bottom-left anchor
    code_array[bottom_anchor_row + (anchor_size // 2), buffer + (anchor_size // 2)] = 1

    # ----------------------------------------------------------------
    # BOTTOM-RIGHT ANCHOR: placed immediately after the last data row, right-aligned
    right_anchor_col = output_size - buffer - anchor_size
    for r in range(anchor_size):
        for c in range(anchor_size):
            # Draw black border for the bottom-right anchor
            if r in [0, anchor_size - 1] or c in [0, anchor_size - 1]:
                code_array[bottom_anchor_row + r, right_anchor_col + c] = 1
    # Small black dot in the center of the bottom-right anchor
    code_array[bottom_anchor_row + (anchor_size // 2), right_anchor_col + (anchor_size // 2)] = 1

    # Convert 1=black => pixel 0; 0=white => pixel 255
    code_image = (1 - code_array) * 255
    return Image.fromarray(code_image.astype('uint8'), 'L')

# Usage
if __name__ == "__main__":
    data = "Good evening humans skibidi ohio rizz mcdonalds"
    code_img = tsimple_custom_code_triangle_anchor(data)
    code_img.save("my_custom_code.png")
    code_img.show()
