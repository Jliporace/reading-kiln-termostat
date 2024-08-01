
import sys
sys.path.append('/home/jessica/reading-kiln-termostat/src')
import InputReader
import TesseractPipeline
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract


# Teste 1

# Cortar a imagem manualmente, cores normais na imagem preta e branco, mascara. Resultado correto na de cima, 
# porém não na debaixo. Tentei blur na debaixo mas também sem sucesso. Replicar teste
test_pb = "/home/jessica/reading-kiln-termostat/notebooks/kiln-images/color-images/12_4000.jpg"
image_pb = cv2.imread(test_pb)

image_pb_nc = image_pb[100:, 0:]
display_sup = image_pb[510:600, 830:1000]
display_inf = image_pb[580:635, 830:1000] 

plt.imshow(display_sup)
plt.show()

plt.imshow(display_inf)
plt.show()

 #valor em BGR
msk = cv2.inRange(image_pb_nc, np.array([250, 250, 250]), np.array([255, 255, 255]))
msk_sup = cv2.inRange(display_sup, np.array([250, 250, 250]), np.array([255, 255, 255]))
msk_inf = cv2.inRange(display_inf, np.array([250, 250, 250]), np.array([255, 255, 255]))
# msk = cv2.inRange(image, np.array([0, 0, 175]), np.array([179, 255, 255]))
# blur = cv2.GaussianBlur(msk,(5,5),10)

plt.imshow(msk_sup)
plt.show()

plt.imshow(msk_inf)
plt.show()

possible_configs = ["--psm 3 -c tessedit_char_whitelist=0123456789",
                    "--psm 4 -c tessedit_char_whitelist=0123456789",
                    "--psm 5 -c tessedit_char_whitelist=0123456789",
                    "--psm 6 -c tessedit_char_whitelist=0123456789",
                    "--psm 7 -c tessedit_char_whitelist=0123456789",
                    "--psm 3",
                    "--psm 4",
                    "--psm 5",
                    "--psm 7",
                    "--psm 6",
                   "--psm 13",
                   "--psm 8"]

for config in possible_configs:
        print("config: ")
        print(config)
        print(pytesseract.image_to_string(msk, config=config))


# Teste 2 - Grayscale. Leitura boa do display superior e inferior ainda impreciso. Lendo 12049 quando 1204
test_color = "/home/jessica/reading-kiln-termostat/notebooks/kiln-images/color-images/0_500.jpg"
test_pb = "/home/jessica/reading-kiln-termostat/notebooks/kiln-images/color-images/12_4000.jpg"

display_sup = image_pb[510:600, 830:1000]
display_inf = image_pb[580:635, 840:990] 
plt.imshow(display_sup, cmap = 'Greys')
plt.show()

plt.imshow(display_inf, cmap = 'Greys')
plt.show()

image_color = cv2.imread(test_color, cv2.IMREAD_GRAYSCALE)
image_pb = cv2.imread(test_pb, cv2.IMREAD_GRAYSCALE)

msk = cv2.inRange(display_inf, np.array([195]), np.array([505]))
bord = cv2.copyMakeBorder(src=msk, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT) 
plt.imshow(bord, cmap = 'Greys')
plt.show()