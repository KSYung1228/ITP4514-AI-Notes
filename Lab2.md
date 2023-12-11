# Lab2 - Collection Data Types, NumPy & Pandas

## Collection Data Types
### List
 - List Items
   - List items are ordered and changeable
   - List items are indexed, the first item has **index[0]**, second item has **index[1]** and last item can use **index[-1]** to represent.
   - The list is changeale, meaning that we can cange, add, and remove items in a list after it has been created.
```py
# list use []
mix = [1,True,'Alan'] # list items can be of ant data type
items = ['a','b','c'] 
items.append("d") # add d to items list
items.remove("a") # remove item a from items list
item = items.pop(2) # remove item from items list position 2, and assign to variable item
items.insert(0,"a") # insert a to index 0

items2 = ['d','e','f'] # new list items2
items.extend(items2) # add items2 to items list

'''
Create List by range()
'''
arr = list(range(1,10))
print(arr)
'''
output -> [1,2,3,4,5,6,7,8,9]
'''

len(arr) #get the length of arr

'''
check sth in list
'''
if xxx in arr:
    print("Y")
    
'''
loop and print items in list
'''
for i in arr:
    print(i)
```

### Tuple
 - Tuples are used to store multiple items in a single variable.
 - A tuple is a acollection chich is ordered and **unchangeable**.
 - Tuples are written with round brackets.
 - Access tuple items methods are same as list
```py
#tuple use ()
tuplea = tuple(items) # convert list to tuple
```

## Numerical Python - NumPy
### Create an array
 - need to install Numpt library for local tools.
 - import NumPy lib
 - Use NumPy method to create an arrays(originally it is a list)
```py
import numpy as np
cars = np.array(['Honda','Toyota','BMW','Merz','Mazda','Proton'])
```
 - can use numpy method arrange(start, stop, step) to create an array
```py
import numpy as np
arr = np.arrange(10)
```
 - Dimension
```py
arr = np.array([0,1,2,3,4,5,6,7]) # 1D arrays
arr2 = np.array([[0,1,2,3],[4,5,6,7]]) # 2D array
```
 - Show shape
```py
print(arr2.shape)#show the shape of array2
```
 - Reshape array
```py
arr3 = arr1.reshape(2,4) # reshape 1D to 2D array with 2x4
```
 - Searching
```py
#Show the location og text
a = np.where(arr == 2)
print(a)
```
 - Sort & Search
```py
arr = np.array[2,5,1,6,9,4,1]
sorted = np.sort(arr)
search = np.searchsorted(sorted, 5)
```
 - filtering an array
```py
filter = [False,True,False,True,False,True,False]
filterarr = arr[filter]
#----------------------
filter = arr>3
filterarr = arr[filter]
```
 - Delete items in array
```py
#Delete from position
arr = np.delete(arr, 3)
#Delete in multiple position
arr = np.delete(arr,[1,4,5])
```
## "Panel Data", and "Python Data Analysis" - Pandas
### Pandas Library
 - Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with "**relational**" or "**labeled**" data both easy and intuitive.
 - Aims to be the fundamental hight-level building block for doing paractial, real world data analysis in Python.
 - Broad goal of becoming the most powerful and flexible open source **data analysys / manipulation tool** available in any language.
 - Two primary data structures of pandas, **Series**(1-dimensional) and **DataFrame**(2-dimensional).
 - Handle the vast majourity of typical use cases in **finance**, **statistics**, **social science**, and many areas of **engineering**

### Series
 - One-dimansional aray-like object containing:
   - an array of data, and
   - an associated array of data labels.
 - Create a Series with an index
```py
import pandas as pd
popul = pd.Series([13.78,3.24,0.65,1.27], index = ['China','USA','UK','Japan'])
print(popul) # show the Series
print(popul['USA']) # show the info of 'USA(index)'
 ```
 - Create a Series with a dictionary:
```py
populdict = {"China":13.78,"USA":3.24,"UK":0.65}
popul = pd.Series(populdict)
print(popul)
```
 - Access Series with indexes:
```py
import pandas as pd
popul = pd.Series([13.78.3.24.0.65.1.27],index = ['China','USA','UK','Japan'])
print(popul[popul<3])
print()
print('HK' in popul)
print()
print(popul[['China','UK']])
```
 - Remove rows in Series bu using drop:
```py
import pandas as pd
objs = pd.Series([5,8,-4,2],index=['a','b','c','d'])
print(objs)
newobjs = objs.drop('b')
```
### DataFrame
 - A DataFrame represents a tabular, speadsheet-like data strcture.
 - It contains an ordered collection of columns, each of which can be a different value type(numeric, string, bolean, etc.).
 - It has both a row and column index
 - There are numerous ways to construct a DataFrame.
 - One of the most common is from a dict of equal-length lists or NumPy arrays
```py
import pandas as pd
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year':[2017,2018,2019,2018,2019],
        'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = pd.DataFrame(data)
print(frame)
print()

print(len(frame.index)) # length of the Data Frame
print()

print(frame.columns) # get column names
print()

print(frame.state) # get column "state"
print()

print(frame.loc[2]) # get the third row
```