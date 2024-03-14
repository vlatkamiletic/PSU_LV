try:
    broj = float(input("Unesi broj: "))
    if broj < 0.0 or broj > 1.0:
        print("Broj je izvan intervala.")
    #elif broj > 0.0 or broj < 1.0 :  
    else:
        if broj >= 0.9:
            print('A')
        elif broj >= 0.8:
            print('B')
        elif broj >= 0.7:
            print('C')
        elif broj >= 0.6:
            print('D')
        elif broj < 0.6:
            print('F')
except:
    print("Nije unesen broj.")