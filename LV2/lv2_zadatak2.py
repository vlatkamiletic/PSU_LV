import numpy as np
import matplotlib.pyplot as plt

# Učitavanje podataka
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1, 2, 3, 4, 5, 6), delimiter=",", skiprows=1)

# Provjera je li data prazan niz
if data.size == 0:
    print("Nema podataka učitanih iz datoteke.")
else:
    # Prikažite ovisnost potrošnje automobila (mpg) o konjskim snagama (hp)
    plt.scatter(data[:, 3], data[:, 0], label='Automobili', c='blue', alpha=0.5)

    # Na istom grafu prikažite i informaciju o težini pojedinog vozila
    # Veličina točkice neka bude u skladu sa težinom wt
    plt.scatter(data[:, 3], data[:, 0], s=data[:, 5]*10, label='Težina', c='red', alpha=0.5)

    # Izračunajte minimalne, maksimalne i srednje vrijednosti potrošnje (mpg) automobila
    if data.size > 0:
        mpg_min = np.min(data[:, 0])
        mpg_max = np.max(data[:, 0])
        mpg_mean = np.mean(data[:, 0])
        print("Minimalna vrijednost mpg:", mpg_min)
        print("Maksimalna vrijednost mpg:", mpg_max)
        print("Srednja vrijednost mpg:", mpg_mean)
    else:
        print("Nema podataka za izračun minimalne, maksimalne i srednje vrijednosti mpg.")

    # Prikazivanje legende i dodavanje oznaka
    plt.xlabel('Konjske snage (hp)')
    plt.ylabel('Potrošnja (mpg)')
    plt.title('Ovisnost potrošnje automobila o konjskim snagama i težini')
    plt.legend()
    plt.grid(True)
    plt.show()
