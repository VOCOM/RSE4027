## 
# Changelog:
# - 17/11/23
#   Created utility library
##

import numpy

def Str2NaN(value):
    if value == "0":
        value = numpy.nan
    return value
