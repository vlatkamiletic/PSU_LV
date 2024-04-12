import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')

#1. Kojih 5 automobila ima najveću potrošnju? (koristite funkciju sort)
print(mtcars.sort_values(by = ['mpg']).head(5))

#2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
print(mtcars[mtcars.cyl == 8].sort_values(by = ['mpg'], ascending = False).head(3))

#3. Kolika je srednja potrošnja automobila sa 6 cilindara?
print(mtcars[mtcars.cyl == 6].mpg.mean())

#4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?
print(mtcars[(mtcars.cyl == 4) & (mtcars.wt > 2) & (mtcars.wt < 2.2)].mpg.mean())

#5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?
print(mtcars.am.value_counts())

#6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?
print(len(mtcars[(mtcars.am == 0) & (mtcars.hp > 100)]))

#7. Kolika je masa svakog automobila u kilogramima?
mtcars['kg'] = mtcars.wt*1000*0.453
print(mtcars[['car','kg']])