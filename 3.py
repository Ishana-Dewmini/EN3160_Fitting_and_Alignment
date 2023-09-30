import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to get the mouse click coordinates
def get_mouse_click(event, x, y, flags, param):
    global selected_points, current_point_index

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points[current_point_index] = (x, y)
        current_point_index += 1
        if current_point_index == 4:
            cv2.destroyAllWindows()

# Load the images
im1 = cv2.imread("tv.jpg")  
im2 = cv2.imread("ocean.jpg")  
# Define a resize factor to make image1 manageable for point selection
resize_factor = 0.25
im1_resized = cv2.resize(im1, None, fx=resize_factor, fy=resize_factor)

# Display the resized image1 using OpenCV for point selection
cv2.imshow('Image1', im1_resized)
cv2.setMouseCallback('Image1', get_mouse_click)

# Initialize variables for selected points
selected_points = [None] * 4
current_point_index = 0

# Wait for the user to select the four corner points by clicking on the resized Image1
print("Please select four points on Image1 for homography calculation.")
print("Click on the bottom-left corner of the object and proceed anti-clockwise.")
while current_point_index < 4:
    cv2.waitKey(1)

# Close the OpenCV window
cv2.destroyAllWindows()

# Display image2 using Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(im1_resized, cv2.COLOR_BGR2RGB))
plt.title("Image1")
plt.axis('off')  # Remove axis labels

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.title("Image2")
plt.axis('off')  # Remove axis labels

# Convert selected points to a NumPy array: order of points is [bottom_left, bottom_right, top_right, top_left]
selected_points = np.array(selected_points, dtype=np.float32)
print(selected_points)

if selected_points.shape == (4, 2):
    # Define the corner points of image2 # shape (445, 894, 3)
    # [bottom_left, bottom_right, top_right, top_left]
    pts1 = np.float32([[0, im2.shape[0]], [im2.shape[1], im2.shape[0]], [im2.shape[1], 0], [0, 0]])
                        

    # Create a transformation matrix to map image2 to image1
    h, _ = cv2.findHomography(pts1,selected_points)

    # Warp image2 to image1
    img2_warp = cv2.warpPerspective(im2, h, (im1_resized.shape[1], im1_resized.shape[0]))

    # Create a mask for the image2 region
    img2_mask = np.zeros_like(im1_resized, dtype=np.uint8)
    cv2.fillConvexPoly(img2_mask, selected_points.astype(int), (255, 255, 255))

    # Invert the mask to get the non-image2 region
    non_img2_mask = cv2.bitwise_not(img2_mask)

    # Remove the image2 region from image1
    img1_without_img2 = cv2.bitwise_and(im1_resized, non_img2_mask)

    # Combine the image2 region and the image1 without the image2 region
    result = cv2.bitwise_or(img2_warp, img1_without_img2)

    # Display the result
    plt.figure()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Image1 with Image2")
    plt.axis('off')  # Remove axis labels

    '''# Visualize the selected points on both images
    plt.plot(selected_points[:, 0], selected_points[:, 1], 'ro')  # Red circles for selected points on image1
    plt.plot(pts1[:, 0], pts1[:, 1], 'go')  # Green circles for corresponding points on image2

    # Draw lines connecting the selected points for visualization
    for i in range(4):
        plt.plot([selected_points[i, 0], pts1[i, 0]], [selected_points[i, 1], pts1[i, 1]], 'b-')'''

    plt.show()
else:
    print("Please select four points on Image1 for homography calculation.")
