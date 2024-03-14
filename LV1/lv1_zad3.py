brojevi = []

while (1):
    num = input("Unesi broj: ")
    if num == "Done":
        break
    try: 
        broj = float(num)
        brojevi.append(broj)

    except ValueError:
        print("Neispravan unos")

print("Uneseno je: ", len(brojevi), "brojeva")

print("Prosjecna vrijednost: ", sum(brojevi)/len(brojevi))
print("Minimalna vrijednost:", min(brojevi))
print("Maksimalna vrijednost:", max(brojevi))



brojevi.sort()
print("Sortirana lista: ", brojevi)