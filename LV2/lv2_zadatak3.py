import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Učitavanje slike
img = mpimg.imread('tiger.png')

# a) Posvijetliti sliku 
brightened_img = np.clip(img * 1.5, 0, 1) 

# b) Rotiranje slike za 90 stupnjeva u smjeru kazaljke na satu
rotated_img = np.rot90(img)

# c) Zrcaliti sliku
flipped_img = np.flipud(img)

# d) Smanjite rezoluciju slike x puta (npr. 10 puta)
downsampled_factor = 10
downsampled_img = img[::downsampled_factor, ::downsampled_factor]

# e) Prikazite samo drugu četvrtinu slike po širini, a prikažite sliku cijelu po visini; ostali dijelovi slike trebaju biti crni
height, width, _ = img.shape
cropped_img = np.zeros_like(img)  # Crna slika istih dimenzija kao original
cropped_width = width // 2  # Pola širine
cropped_img[:, cropped_width:width, :] = img[:, cropped_width:width, :]  # Postavljamo drugu polovicu slike

# Prikazivanje originalne slike i manipuliranih slika
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Originalna slika')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(brightened_img)
plt.title('Posvijetljena slika')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(rotated_img)
plt.title('Rotirana slika')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(flipped_img)
plt.title('Zrcaljena slika')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(downsampled_img)
plt.title('Smanjena rezolucija')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cropped_img)
plt.title('Ograničena slika')
plt.axis('off')

plt.tight_layout()
plt.show()