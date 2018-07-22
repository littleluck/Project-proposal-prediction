import numpy as np 
import pandas as pd

df1 = pd.DataFrame({"id":[1,2,3]})
df2 = pd.DataFrame({"id":[4,5,6]})
for x in [df1, df2]:
    print(x["id"])


