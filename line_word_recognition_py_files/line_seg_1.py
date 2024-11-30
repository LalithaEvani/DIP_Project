import cv2
import numpy as np
# import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import os

def crop_and_save_lines(text_image, lines_binary_image, output_folder):
  
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find connected components in the binary lines image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lines_binary_image.astype(np.uint8), connectivity=8)
    # print(stats[:,2])
    # Get the maximum width of all the lines
    max_line_width = 0
    for stat in stats[1:]:  # Skip the background label
        max_line_width = max(max_line_width, stat[2])  # Width is the 3rd element in 'stat'
    
    # Iterate through each line component
    stats_n=[]
    for i in range(1, num_labels):  # Skip the background label (0)
        x, y, w, h, _ = stats[i]

        if w> lines_binary_image.shape[1]/6:
            stats_n.append(stats[i])

    print('number of lines detected:',len(stats_n))
    min_line_width = text_image.shape[0]/(len(stats_n)*4)
    for i in range(len(stats_n)):

        # if i!=2:
        #     continue

        x1, y1, w1, h1, _ = stats_n[i]
        if i==(len(stats_n)-1):
            x2, y2, w2, h2 =0, text_image.shape[0],text_image.shape[1],1
        else:
            x2, y2, w2, h2, _ = stats_n[i+1]


        # print(stats_n[i],[x2, y2, w2, h2])
 
        cropped_line = text_image[y1:y2+h2, :].copy()
        line_mask = lines_binary_image[y1:y2+h2, :].copy()
        if i==(len(stats_n)-1):
            line_mask[-1]=1
        # plt.imshow(line_mask)
        # plt.show()
        # print(line_mask.shape)
        # cv2.bitwise_not(skeleton)
        for c in range(line_mask.shape[1]):
            s_p=None
            e_p =0
            for j in range(line_mask.shape[0]):
                if s_p == None and line_mask[j,c]==1:
                    s_p= j
                    # print('sp',j)
                if s_p != None and line_mask[j,c]==1 and j>(s_p+min_line_width):
                    e_p=j
                    # print(j)
            line_mask[s_p:e_p,c]=1
        cropped_line_n = cropped_line*line_mask
        # plt.imshow(line_mask)
        # plt.show()
        # print(cropped_line_n.shape)
        # plt.imshow(cv2.bitwise_not(cropped_line_n), cmap='gray')
        # plt.show()
        # if i==2:
        #     break
        # line_image_path=output_folder+f'{image_name}_{i}.png'
        line_image_path = os.path.join(output_folder, f'{i}.png')
        # line_image_path=output_folder+f'{i}.png'
        f = cv2.imwrite(line_image_path, cv2.bitwise_not(cropped_line_n))
        # print(f"{f}_Saved: {line_image_path}")

# def find_endpoints(skeleton):

#     kernel = np.array([[1, 1, 1],
#                        [1, 10, 1],
#                        [1, 1, 1]], dtype=np.uint8)
    
#     # Convolution to find neighbors of each pixel
#     neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
#     # Identify endpoints (pixels with exactly one neighbor)
#     endpoints = np.where((neighbors == 11) & (skeleton == 1))
    
#     # Convert to list of (x, y) coordinates
#     endpoints_n = list(zip(endpoints[1], endpoints[0]))
#     # print('Before filtering:', endpoints_n)
    
#     # Filter endpoints based on their x-coordinate
#     endpoints_filtered = [endpoint for endpoint in endpoints_n if endpoint[0] > (skeleton.shape[0] / 10)]
    
#     # Determine the direction of each endpoint
#     endpoints_with_direction = []
#     for endpoint in endpoints_filtered:
#         x, y = endpoint
        
#         # Check 8-connected neighbors
#         neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
#                             (0, -1),         (0, 1),
#                             (1, -1), (1, 0), (1, 1)]
#         neighbor_offsets=1*neighbor_offsets
#         neighbors = []
#         for dx, dy in neighbor_offsets:
#             nx, ny = x + dx, y + dy
#             if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] == 1:
#                 neighbors.append((nx, ny))
        
#         # Determine the direction based on the position of the neighbor
#         if neighbors:
#             # print(neighbors)
#             # Calculate the vector between the endpoint and the neighbor
#             neighbor = neighbors[0]  # Take the first neighbor (assumes a single path)
#             dx = x-neighbor[0] 
#             dy = y-neighbor[1]
            
#             # Determine direction
#             if dx<0:  # Horizontal component dominates
#                 direction = 0 #'right'
#             else:  # Vertical component dominates (for future modifications if needed)
#                 direction = 1 #'left'
#         else:
#             # print('No neighbours')
#             direction = 0 #'undefined'  # Fallback if no valid neighbors are found
        
#         endpoints_with_direction.append((x, y, direction))
    
#     # print('After filtering with direction:', endpoints_with_direction)
    
#     # Calculate number of removed lines
#     n_lines = len(endpoints_n) - len(endpoints_filtered)
    
#     return endpoints_with_direction, n_lines


# def extend_horizontally(image, x, y, direction, max_length=50):
#     """
#     Extend the line horizontally from (x, y) in the given direction.
#     """
#     for i in range(1, max_length + 1):
#         new_x = x + i * direction  # Extend horizontally
#         if new_x < 0 or new_x >= image.shape[1]:
#             break  # Stop if reaching the boundary
#         if (y+1)<image.shape[0] and (new_x+1)<image.shape[1]:
#             if image[y+1,new_x+1]!=0 or image[y,new_x+1]!=0 or image[y-1,new_x+1]!=0:
#                 if image[y+1,new_x+1]!=0:
#                     y= y-1
#                 if image[y,new_x+1]!=0:
#                     y= y-2
#                 if image[y-1,new_x+1]!=0:
#                     y= y+1

#             else:
#                 image[y, new_x] = 1  # Set pixel value to white


#     return image

# def connect_lines(image, endpoints,n_lines, max_distance):
#     """
#     Connect nearby endpoints or extend lines horizontally if no match is found.
#     """
#     v_threshold = image.shape[0]/(n_lines*(1.2))
#     image_n = image.copy()
#     used = set()  # Track used endpoints
#     for i, (x1, y1,direction) in enumerate(endpoints):
#         if i in used:
#             continue
#         for j, (x2, y2,d) in enumerate(endpoints):
#             if i != j and j not in used:
#                 distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#                 # print('yes',abs(y1 - y2),distance)

#                 if distance < max_distance and abs(y1 - y2) < v_threshold and d==0 and direction==1:  # Allow only horizontal-ish connecti
#                     # print('yes',abs(y1 - y2),distance)
#                     cv2.line(image_n, (x1, y1), (x2, y2), 1, 1)
#                     used.add(i)
#                     used.add(j)
#                     break
#         else:
#             # If no match, extend the line horizontally
#             if direction==1:
#             # direction = 0 if x1 < image_n.shape[1] // 2 else 1  # Left or right
#                 # print(x1,y1,image.shape[1]-x1)
#                 image_n = extend_horizontally(image_n, x1, y1, direction,max_length= image.shape[1]-x1)
#     return image_n




def find_endpoints(skeleton):
 
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolution to find neighbors of each pixel
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Identify endpoints (pixels with exactly one neighbor)
    endpoints = np.where((neighbors == 11) & (skeleton == 1))
    
    # Convert to list of (x, y) coordinates
    endpoints_n = list(zip(endpoints[1], endpoints[0]))
    # print('Before filtering:', endpoints_n)
    
    # Filter endpoints based on their x-coordinate
    endpoints_filtered_n = [endpoint for endpoint in endpoints_n if endpoint[0] > (skeleton.shape[0] / 10)] #edited
    endpoints_filtered = endpoints_n
    
    # Determine the direction of each endpoint
    endpoints_with_direction = []
    for endpoint in endpoints_filtered:
        x, y = endpoint
        
        # Check 8-connected neighbors
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),         (0, 1),
                            (1, -1), (1, 0), (1, 1)]
        neighbor_offsets=1*neighbor_offsets
        neighbors = []
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] == 1:
                neighbors.append((nx, ny))
        
        # Determine the direction based on the position of the neighbor
        if neighbors:
            # print(neighbors)
            # Calculate the vector between the endpoint and the neighbor
            neighbor = neighbors[0]  # Take the first neighbor (assumes a single path)
            dx = x-neighbor[0] 
            dy = y-neighbor[1]
            
            # Determine direction
            if dx<0:  # Horizontal component dominates
                direction = -1 #'right'
            else:  # Vertical component dominates (for future modifications if needed)
                direction = 1 #'left'
        else:
            # print('No neighbours')
            direction = 0 #'undefined'  # Fallback if no valid neighbors are found
        
        endpoints_with_direction.append((x, y, direction))
    
    # print('After filtering with direction:', endpoints_with_direction)
    
    # Calculate number of removed lines
    n_lines = len(endpoints_n) - len(endpoints_filtered_n)
    
    return endpoints_with_direction, n_lines


def extend_horizontally(image, x, y, direction, max_length=50):
    """
    Extend the line horizontally from (x, y) in the given direction.
    """
    for i in range(1, max_length + 1):
        new_x = x + i * direction  # Extend horizontally
        if new_x < 0 or new_x >= image.shape[1]:
            break  # Stop if reaching the boundary

        if direction!=-1:
            if (y-1)>0 and (y+1)<image.shape[0] and (new_x+1)<image.shape[1]:
                if image[y+1,new_x+1]!=0 or image[y,new_x+1]!=0 or image[y-1,new_x+1]!=0:
                    if image[y+1,new_x+1]!=0:
                        y= y-1
                    if image[y,new_x+1]!=0:
                        y= y-2
                    if image[y-1,new_x+1]!=0:
                        y= y+1
                    # print("in -1")

                    # image[y, new_x] = 1 

                else:
                    image[y, new_x] = 1  # Set pixel value to white

        else:
            if (y-1)>0 and (y+1)<image.shape[0] and (new_x-1)>0:
                if image[y+1,new_x-1]!=0 or image[y,new_x-1]!=0 or image[y-1,new_x-1]!=0:
                    if image[y+1,new_x-1]!=0:
                        y= y-1
                    if image[y,new_x-1]!=0:
                        y= y-2
                    if image[y-1,new_x-1]!=0:
                        y= y+1
                    # print("in -1")
                    # image[y, new_x] = 1 
                else:
                    image[y, new_x] = 1  # Set pixel value to white
                    # print("yes -1d")
    return image

def connect_lines(image, endpoints,n_lines, max_distance):
    """
    Connect nearby endpoints or extend lines horizontally if no match is found.
    """
    v_threshold = image.shape[0]/(n_lines*(1.2))
    image_n = image.copy()
    used = set()  # Track used endpoints
    for i, (x1, y1,direction) in enumerate(endpoints):
        if i in used:
            continue
        for j, (x2, y2,d) in enumerate(endpoints):
            if i != j and j not in used:
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # print('yes',abs(y1 - y2),distance)

                if distance < max_distance and abs(y1 - y2) < v_threshold and d==-1 and direction==1:  # Allow only horizontal-ish connecti
                    print('yes',abs(y1 - y2),distance)
                    cv2.line(image_n, (x1, y1), (x2, y2), 1, 1)
                    used.add(i)
                    used.add(j)
                    break
        else:
            # If no match, extend the line horizontally
            if direction==1:
            # direction = 0 if x1 < image_n.shape[1] // 2 else 1  # Left or right
                # print(x1,y1,image.shape[1]-x1)
                image_n = extend_horizontally(image_n, x1, y1, direction,max_length= image.shape[1]-x1)
            if direction==-1 and x1<image_n.shape[1]/5:
                image_n = extend_horizontally(image_n, x1, y1, direction,max_length= x1-1)

    return image_n



def skeletoniz_n(img):
    skel = np.zeros_like(img)
    eroded = np.zeros_like(img)
    temp = np.zeros_like(img)

    while True:
        # Erode the image
        eroded = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        temp = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel

def fill_horizontal_white_gaps(image, gap_threshold):
    filled_image = image.copy()
    rows, cols = filled_image.shape

    for row in range(rows):
        start = None
        for col in range(cols):
            # Identify black regions (0) with white gaps (255)
            if filled_image[row, col] == 0:
                if start is not None:
                    # If a gap is detected between two black regions
                    gap_width = col - start - 1
                    if gap_width > 0 and gap_width <= gap_threshold:
                        # Fill the gap with black (0) if within the threshold
                        filled_image[row, start + 1:col] = 0
                start = col  # Update the start position

    return filled_image

def fill_vertical_gaps(image, gap_threshold, stripe_width):
    filled_image = image.copy()
    rows, cols = filled_image.shape
    # to fill the vertical gaps between two black lines
    for col in range(cols):
        start = None
        for row in range(rows):
            # Identify black regions (0) with white gaps (255)
            if filled_image[row, col] == 0:
                if start is not None:
                    # If a gap is detected between two black regions
                    gap_height = row - start - 1
                    if gap_height > 0 and gap_height <= gap_threshold:
                        # Fill the gap with black (0) if within the threshold
                        filled_image[start + 1:row, col] = 0
                start = row  # Update the start position

    return filled_image


def PPA_Algo_line(image_path, output_folder, image_name, vartical_gap_threshold = 10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate the Average width of components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    component_widths = stats[1:, cv2.CC_STAT_WIDTH]  # Exclude the background
    average_width = np.mean(component_widths)
    print(f"Average width of components: {average_width}")

    stripe_width = int(average_width)
    height, width = blurred.shape
    stripes = [blurred[:, i:i + stripe_width] for i in range(0, width, stripe_width)]

    # paint the stripes
    painted_image = np.zeros_like(blurred)

    for i, stripe in enumerate(stripes):
        for row in range(stripe.shape[0]):
            avg_gray = np.mean(stripe[row, :])
            stripe[row, :] = avg_gray
        painted_image[:, i * stripe_width:(i + 1) * stripe_width] = stripe
    _, painted_binary_image = cv2.threshold(painted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    filled_image = fill_vertical_gaps(painted_binary_image, vartical_gap_threshold,stripe_width)


    # Dilation operation
    threshold_width=4*stripe_width
    sum_s = 1
    filled_image_n=filled_image.copy()
    for i in range(0, filled_image.shape[1], stripe_width):
        for j in range(0, filled_image.shape[0]-threshold_width):
            sum_s = np.sum(filled_image[j:j+threshold_width, i:i + stripe_width])
            if sum_s<=0:
                filled_image_n[j:j+threshold_width, i:i + stripe_width]=255

    gap_threshold = stripe_width*4 


#   Complementing and thinning operations 
    filled_image_n_n = fill_horizontal_white_gaps(filled_image_n, gap_threshold)

    skeleton = skeletoniz_n(filled_image_n_n)
    thinned_background = cv2.bitwise_not(skeleton)


#   Trimming process
    inverted_image = cv2.bitwise_not(thinned_background)
    # Define the horizontal Sobel-like filter
    H = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]], dtype=np.float32)

    # Apply the horizontal filter
    for i in range(4):
        filtered_image = cv2.filter2D(inverted_image, -1, H)
        inverted_image=filtered_image
    # Normalize and threshold the result
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    _, thresholded_image = cv2.threshold(filtered_image, 50, 255, cv2.THRESH_BINARY)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_image, connectivity=8)
    # Get image dimensions
    image_height, image_width = thresholded_image.shape
    # Minimum line length (1/5th of image width)
    min_length = image_width / 5
    # Create an empty mask for the filtered image
    filtered_image = np.zeros_like(thresholded_image)
    # Loop through each component
    for i in range(1, num_labels):  # Skip the background (label 0)
        x, y, w, h, area = stats[i]  # Bounding box and area
        if w > min_length:  # Keep only components with width greater than min_length
            filtered_image[labels == i] = 255

    _, binary_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY)

    # Skeletonize the image
    skeleton = skeletonize(binary_image // 255).astype(np.uint8)

    # Detect endpoints
    endpoints,n_lines = find_endpoints(skeleton)
    # print(endpoints,n_lines)
    # Define maximum connection distance (1/10th of image width)
    image_height, image_width = skeleton.shape
    max_distance = image_width / 10

    # Connect endpoints and extend lines horizontally
    completed_image = connect_lines(skeleton, endpoints, n_lines, max_distance)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    completed_image_c = completed_image.copy()
    crop_and_save_lines(binary_image, completed_image_c, output_folder)

    return painted_binary_image,filled_image_n, filled_image_n_n, filtered_image,completed_image,binary_image+255*completed_image

