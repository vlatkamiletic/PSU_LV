file_name = input("Unesite ime tekstualne datoteke: ")
try:
    with open(file_name, 'r') as file:
        brojac = 0
        total = 0

        for line in file:
            if line.startswith('X-DSPAM-Confidence:'):
                confidence = float(line.split(':')[1])
                total += confidence
                brojac += 1

        if brojac > 0:
            avr_confidence = total / brojac
            print("Srednja vrijednost pouzdanosti je:", avr_confidence)
        else:
            print("Nema linija oblika X-DSPAM-Confidence u datoteci.")
except FileNotFoundError:
    print("Datoteka nije pronadena.")