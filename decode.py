from PIL import Image
import numpy as np
import cv2

def decode_custom_code(image_path):
    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img)

    # Assume the image is perfectly aligned (no rotation or skew).
    # Determine width/height in pixels:
    height, width = img_array.shape
    size = height
    

    # Estimate cell size in pixels (assuming a square overall).
    cell_width = width // size
    cell_height = height // size
    
    # Extract bits
    bits = []
    for r in range(size):
        for c in range(size):
            # Sample the center of the cell
            row_center = int((r + 0.5) * cell_height)
            col_center = int((c + 0.5) * cell_width)
            
            # If the pixel is dark, treat it as 1; otherwise 0
            pixel_value = img_array[row_center, col_center]
            bit = 1 if pixel_value < 128 else 0
            bits.append(bit)
    
    # Convert bits to characters (8 bits per ASCII char)
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break  # incomplete byte
        byte_val = 0
        for b in byte_bits:
            byte_val = (byte_val << 1) | b
        chars.append(chr(byte_val))
    
    return "".join(chars)

def decode_custom_code_triangle(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    size = img_array.shape[0]

    bits = []
    for r in range(size):
        row_cells = r + 1
        offset = (size - row_cells) // 2
        for c in range(row_cells):
            pixel_value = img_array[r, offset + c]
            bit = 1 if pixel_value < 128 else 0
            bits.append(bit)

    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        val = 0
        for b in byte_bits:
            val = (val << 1) | b
        chars.append(chr(val))

    return "".join(chars)

def decode_custom_code_triangle_photo_anchor(image_path):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 1) Find bounding box of dark pixels
    dark_pixels = (img_array < 128)
    rows = np.any(dark_pixels, axis=1)
    cols = np.any(dark_pixels, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_array[rmin:rmax+1, cmin:cmax+1]

    # 2) Detect anchor at the top. This example just skips the first 5 stacked rows.
    anchor_size = 5
    size = cropped.shape[0]
    bits = []
    for r in range(anchor_size, size):
        row_cells = r + 1
        offset = (size - row_cells) // 2
        for c in range(row_cells):
            if 0 <= offset + c < cropped.shape[1]:
                pixel_value = cropped[r, offset + c]
                bit = 1 if pixel_value < 128 else 0
                bits.append(bit)

    # 3) Convert bits to text
    chars = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        val = 0
        for b in chunk:
            val = (val << 1) | b
        chars.append(chr(val))

    return "".join(chars)

def tdecode_custom_code_triangle_photo_anchor(image_path):
    from PIL import Image
    import numpy as np

    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 1) Find bounding box of dark pixels so we isolate the code area
    dark_pixels = (img_array < 128)
    if not np.any(dark_pixels):
        return ""
    rows = np.any(dark_pixels, axis=1)
    cols = np.any(dark_pixels, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]

    # 2) Knowing that our encoder uses two anchors (top and bottom),
    #    skip the top 5 and bottom 5 rows:
    anchor_size = 5
    height, width = cropped.shape
    data_rows = height - 2 * anchor_size  # number of data rows

    # Compute horizontal center.
    # (Since the TOP anchor was centered, we assume the active region is roughly centered.)
    center = width // 2

    bits = []
    # Loop over each data row (row index in the cropped image, offset by top anchor)
    for i in range(data_rows):
        # The effective data row index, where row 0 corresponds to 1 cell, row 1 has 2 cells, etc.
        row_cells = i + 1
        r = anchor_size + i  # actual row in cropped image
        # Center this row horizontally around 'center'
        col_offset = center - (row_cells // 2)
        for c in range(row_cells):
            col = col_offset + c
            if col < 0 or col >= width:
                continue
            # For robustness (in case of noise from a phone image), sample a small 3x3 block
            r_min = max(r - 1, 0)
            r_max = min(r + 2, height)
            c_min = max(col - 1, 0)
            c_max = min(col + 2, width)
            block = cropped[r_min:r_max, c_min:c_max]
            avg = np.mean(block)
            bit = 1 if avg < 128 else 0
            bits.append(bit)

    # 3) Convert the collected bits into text (8 bits per ASCII character)
    chars = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        val = 0
        for b in chunk:
            val = (val << 1) | b
        chars.append(chr(val))
    return "".join(chars)

def decode_custom_code_triangle_photo_triple_anchor(image_path, model_w=None, model_h=None):
    """
    An updated decoder that visualizes anchor detection and uses a perspective transform.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Step 1: Load & Otsu threshold image
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise ValueError("Image not found: " + image_path)
    _, thresh = cv2.threshold(orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("ttest.png", thresh)


    # Step 2: Find candidate anchors (quadrilateral contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = orig.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("detected_contours2.png", contour_img)
    # cv2.imwrite("cont.png", contours)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Adjust these thresholds as needed
        if area < 10 or area > 50000:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            candidates.append(approx)
    if len(candidates) < 3:
        raise ValueError("Could not detect enough anchor candidates. Detected: " + str(len(candidates)))

    # Compute candidate centers for debugging
    anchor_centers = []
    for cnt in candidates:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        anchor_centers.append((cx, cy))
    if len(anchor_centers) < 3:
        raise ValueError("Not enough valid anchor centers found.")
        
    # Debug: draw detected centers on a copy and save for inspection
    debug_img = cv2.cvtColor(orig.copy(), cv2.COLOR_GRAY2BGR)
    for pt in anchor_centers:
        cv2.circle(debug_img, pt, 5, (0,0,255), -1)
    cv2.imwrite("detected_anchors.png", debug_img)

    # Step 3: Identify the three anchors.
    # Assume the top anchor has smallest y.
    anchor_centers.sort(key=lambda pt: pt[1])
    top_anchor = anchor_centers[0]
    bottom_candidates = anchor_centers[1:]
    bottom_candidates.sort(key=lambda pt: pt[0])
    bottom_left = bottom_candidates[0]
    bottom_right = bottom_candidates[-1]

    # Step 4: Synthesize a fourth point for perspective transform.
    # We have three anchors from encoder. For full rectification, let’s compute an expected fourth.
    # For example, assume the fourth (missing) anchor is at the bottom, roughly aligned
    # with the top anchor and bottom-left/right anchors.
    # Here we compute the intersection of the line through bottom anchors and the vertical line through top_anchor.
    H, W = orig.shape
    synth_x = top_anchor[0]
    synth_y = bottom_left[1]  # or average bottom y values
    fourth_pt = (synth_x, synth_y)

    # For our model, we define canonical positions.
    buffer_val = 2
    anchor_size = 5
    if model_w is None or model_h is None:
        model_w = W  # or set a fixed size (e.g., 300)
        model_h = H
    # We map:
    # top_anchor -> (model_w/2, buffer_val+anchor_size/2)
    # bottom_left -> (buffer_val+anchor_size/2, model_h - buffer_val - anchor_size/2)
    # bottom_right -> (model_w - buffer_val - anchor_size/2, model_h - buffer_val - anchor_size/2)
    # fourth (synthesized) -> (model_w/2, model_h - buffer_val - anchor_size/2)
    dst_pts = np.float32([
        [model_w/2, buffer_val + anchor_size/2], 
        [buffer_val + anchor_size/2, model_h - buffer_val - anchor_size/2],
        [model_w - buffer_val - anchor_size/2, model_h - buffer_val - anchor_size/2],
        [model_w/2, model_h - buffer_val - anchor_size/2]
    ])
    src_pts = np.float32([top_anchor, bottom_left, bottom_right, fourth_pt])
    
    # Step 5: Compute perspective transform and warp the image.
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(orig, M_persp, (int(model_w), int(model_h)))
    cv2.imwrite("aligned.png", warped)

    # Step 6: Decode the data region.
    data_top = int(buffer_val + anchor_size)
    data_bottom = int(model_h - buffer_val - anchor_size)
    tri_size = data_bottom - data_top  # number of data rows
    bits = []
    for i in range(tri_size):
        row_cells = i + 1
        r = data_top + i
        col_offset = int(model_w/2 - (row_cells // 2))
        for c in range(row_cells):
            col = col_offset + c
            if col < 0 or col >= model_w:
                continue
            r_min = max(r - 1, 0)
            r_max = min(r + 2, int(model_h))
            c_min = max(col - 1, 0)
            c_max = min(col + 2, int(model_w))
            block = warped[r_min:r_max, c_min:c_max]
            avg = np.mean(block)
            bit = 1 if avg < 128 else 0
            bits.append(bit)

    chars = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        val = 0
        for b in chunk:
            val = (val << 1) | b
        chars.append(chr(val))
    return "".join(chars)


if __name__ == "__main__":
    decoded_data = decode_custom_code_triangle_photo_triple_anchor("my_custom_code.png")
    print("Decoded Data:", decoded_data)


# Usage (assuming you used the same generator code):
if __name__ == "__main__":
    decoded_data = decode_custom_code_triangle_photo_triple_anchor("my_custom_code.png")
    # decoded_data = decode_triangle_from_phone_image("phoneimg.jpg")
    print("Decoded Data:", decoded_data)





# Usage (assuming you used the same generator code):
if __name__ == "__main__":
    decoded_data = decode_custom_code_triangle_photo_triple_anchor("my_custom_code.png")
    # decoded_data = decode_triangle_from_phone_image("phoneimg.jpg")
    print("Decoded Data:", decoded_data)
