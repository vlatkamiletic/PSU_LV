import requests
import pandas as pd
from datetime import datetime

# Funkcija za dohvaćanje podataka o kvaliteti zraka za određenu godinu i grad
def get_air_quality_data(year, city):
    url = f"http://iszz.azo.hr/iskzl/rest/ispitivanjapodataka/getAll?p_godina={year}&p_grad={city}&_={int(datetime.now().timestamp())}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items']
    else:
        print("Greska.")
        return None

#Dohvacanje podataka za 2017. godinu za grad Osijek
data = get_air_quality_data(2017, "Osijek")

if data:
    #Pretvaranje podataka u DataFrame
    df = pd.DataFrame(data)
    
    #Pretvaranje stupca "Datum" u format datuma
    df['Datum'] = pd.to_datetime(df['Datum'])

    #1. Dohvacanje mjerenja dnevne koncentracije lebdecih cestica PM10 za 2017. godinu za grad Osijek
    pm10_osijek_2017 = df[(df['Datum'].dt.year == 2017) & (df['Grad'] == 'Osijek') & (df['Parameter'] == 'PM10')]
    print("Mjerenja dnevne koncentracije lebdecih cestica PM10 za 2017. godinu za grad Osijek:")
    print(pm10_osijek_2017)

    #2. Ispis tri datuma u godini kada je koncentracija PM10 bila najveća
    top_3_dates = pm10_osijek_2017.nlargest(3, 'Vrijednost')['Datum']
    print("\nTri datuma u 2017. godini kada je koncentracija PM10 bila najveca su:")
    print(top_3_dates)