import py_compile
import os

files = os.listdir('./source/')

for f in files:
    if f.split('.')[-1]=='py':
        py_compile.compile('./source/'+f)
        


