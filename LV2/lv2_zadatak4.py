import numpy as np
import matplotlib.pyplot as plt

def sahovnica(dim_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina):
    # Dimenzije slike
    visina = broj_kvadrata_visina * dim_kvadrata
    sirina = broj_kvadrata_sirina * dim_kvadrata

    # Kreiranje crnog i bijelog polja
    crno_polje = np.zeros((dim_kvadrata, dim_kvadrata))
    bijelo_polje = np.ones((dim_kvadrata, dim_kvadrata))

    # Kreiranje paterna sahovnice
    polje = np.zeros((visina, sirina))

    for i in range(broj_kvadrata_visina):
        for j in range(broj_kvadrata_sirina):
            if (i + j) % 2 == 0:
                polje[i*dim_kvadrata:(i+1)*dim_kvadrata, j*dim_kvadrata:(j+1)*dim_kvadrata] = bijelo_polje
            else:
                polje[i*dim_kvadrata:(i+1)*dim_kvadrata, j*dim_kvadrata:(j+1)*dim_kvadrata] = crno_polje

    # Dodavanje okvira oko slike
    okvir = np.zeros((visina+2, sirina+2))
    okvir[1:-1, 1:-1] = polje

    # Ispisivanje vrijednosti na osima
    plt.xticks(np.arange(0, sirina+1, dim_kvadrata), np.arange(0, sirina+1, dim_kvadrata))
    plt.yticks(np.arange(0, visina+1, dim_kvadrata), np.arange(0, visina+1, dim_kvadrata))

    # Postavljanje y osi od gore prema dolje
    y_labels = np.arange(0, 176, 25)
    y_positions = np.linspace(visina, 0, len(y_labels))
    plt.yticks(y_positions, y_labels)

    # Prikazivanje slike
    plt.imshow(okvir, cmap='gray', vmin=0, vmax=1)
    plt.show()

# Testiranje funkcije
dim_kvadrata = 50
broj_kvadrata_visina = 4
broj_kvadrata_sirina = 5

sahovnica(dim_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina)
