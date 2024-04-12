'''
import numpy as np
a = np.array([6, 2, 9]) #napravi polje od tri elementa
print(type(a)) #prikaži tip polja
print(a.shape) #koliko elemenata ima vektor
print(a[0], a[1], a[2]) #prikaži prvi, drugi i treći element
a[1] = 5 #promijeni vrijednost polja na drugom mjestu
print(a) #prikaži cijeli a
print(a[1:2]) #izdvajanje
print(a[1:-1]) #izdvajanje
b = np.array([[3,7,1],[4,5,6]]) #napravi 2 dimenzionalno polje (matricu)
print(b.shape) #ispiši dimenzije polja
print(b) #ispiši cijelo polje b
print(b[0, 2], b[0, 1], b[1, 1]) #ispiši neke elemente polja
print(b[0:2,0:1]) #izdvajanje
print(b[:,0]) #izdvajanje
c = np.zeros((4,2)) #polje sa svim elementima jednakim 0
d = np.ones((3,2)) #polje sa svim elementima jednakim 1
e = np.full((1,2),5) #polje sa svim elementima jednakim 5
'''

#primjer vizualizacije
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 6, num=30)
y= np.sin(x)
plt.plot(x, y, 'b', linewidth=1, marker=".", markersize=5)
plt.axis([0,6,-2,2])
plt.xlabel('x')
plt.ylabel('vrijednosti funkcije')
plt.title('sinus funkcija')
plt.show()