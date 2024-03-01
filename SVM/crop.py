

import cv2
import glob

# Get a list of all the image and mask files in the "melanome" directory
image_files = sorted(glob.glob('/home/mohamed/Desktop/Skin-Cancer-detection/BDD_ISIC_2019_NORM/MELANOME/*.JPG'))
mask_files = sorted(glob.glob('/home/mohamed/Desktop/Skin-Cancer-detection/BDD_ISIC_2019_NORM/MELANOMEMASK/*.jpg'))

# Loop over the image-mask pairs
for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
    print(f"Processing image {i + 1}/{len(image_files)}: {image_file}")

    # Load the image and mask
    img = cv2.imread(image_file)
    mask = cv2.imread(mask_file, 0)
    
    # Resize the image to the size of the mask (256x256)
    img = cv2.resize(img, (256, 256))


    # Convert the binary mask to the correct type
    mask = cv2.convertScaleAbs(mask)

    # Verify that the binary mask is the same size as the source image
    assert img.shape[:2] == mask.shape[:2], "Error: Mask size does not match image size"

 
    # Multiply the image with the binary mask
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Find the coordinates of the non-zero elements of the binary mask
    non_zero_coords = cv2.findNonZero(mask)

    # Crop the image based on the coordinates
    x, y, w, h = cv2.boundingRect(non_zero_coords)
    cropped_img = masked_img[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite(f"/home/mohamed/Desktop/Skin-Cancer-detection/BDD_ISIC_2019_NORM/MELANOMESEG/cropped_image_{i + 1}.jpg", cropped_img)

print("Done!")
