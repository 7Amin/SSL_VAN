import cv2
import random

# urls = ["input_slide_2_channel_0.png", "input_slide_3_channel_0.png", "input_slide_4_channel_0.png",
#         "input_slide_2_channel_3.png", "input_slide_3_channel_3.png", "input_slide_4_channel_3.png"]


# for url in urls:
#     image = cv2.imread(url, cv2.IMREAD_COLOR)
#     x, y, z = image.shape
#     image[:, y // 3, :] = (0, 0, 255)
#     image[:, 2 * y // 3, :] = (0, 0, 255)
#
#     image[x // 3, :, :] = (0, 0, 255)
#     image[2 * x // 3, :, :] = (0, 0, 255)
#
#     cv2.imwrite(url[:-4] + "_test" + ".png", image)

URLs = [2, 3, 4, 5, 9, 10, 11, 12]
base = "./new_images_mask/"

patches = [2, 4, 8, 16, 32]

for number in URLs:
    url = base + str(number) + ".png"
    c = random.randint(0, 4)
    print(c)
    p = patches[c]
    img = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
    x, y = img.shape
    xx = x // p + 1
    yy = y // p + 1
    q = 75
    for i in range(xx):
        for j in range(yy):
            r = random.randint(0, 100)
            if r <= q:
                img[i * p: min((i + 1) * p, x), j * p: min((j + 1) * p, y)] = 1.0
    print(f"x is {x} and y is {y}")
    cv2.imwrite(base + str(number) + "_masked" + ".png", img)

