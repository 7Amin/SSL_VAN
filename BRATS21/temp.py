import cv2

urls = ["input_slide_2_channel_0.png", "input_slide_3_channel_0.png", "input_slide_4_channel_0.png",
        "input_slide_2_channel_3.png", "input_slide_3_channel_3.png", "input_slide_4_channel_3.png"]


for url in urls:
    image = cv2.imread(url, cv2.IMREAD_COLOR)
    x, y, z = image.shape
    image[:, y // 3, :] = (0, 0, 255)
    image[:, 2 * y // 3, :] = (0, 0, 255)

    image[x // 3, :, :] = (0, 0, 255)
    image[2 * x // 3, :, :] = (0, 0, 255)

    cv2.imwrite(url[:-4] + "_test" + ".png", image)
