'''
Created on Dec 26, 2014

@author: edwingsantos
'''

import matplotlib.pyplot as plt
import scipy as sp

def error(f,x,y):
    return sp.sum((f(x)-y)**2)


data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
x = data[:,0]
y = data[:,1]

sp.sum(sp.isnan(y))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
     ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()


fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
print (residuals)

f1 = sp.poly1d(fp1)
print (error(f1,x,y))

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth = 4)
plt.legend(["d=%i" % f1.order], loc ="upper left")

f2 = sp.polyfit(x,y,2)
#print (f2)

f2 = sp.poly1d(f2)
print(error(f2, x, y))

fx2  = sp.linspace(0, x[-1], 1000)
plt.plot(fx2, f2(fx2), linewidth = 3)
plt.legend(["d=%i" % f2.order], loc ="upper left")

#====
f3 = sp.polyfit(x,y,10)
#print (f3)

f3 = sp.poly1d(f3)
print(error(f3, x, y))

fx3  = sp.linspace(0, x[-1], 1000)
plt.plot(fx3, f3(fx3), linewidth = 3)
plt.legend(["d=%i" % f3.order], loc = "upper left")


f4 = sp.polyfit(x,y,100)
#print (f4)

f4 = sp.poly1d(f4)
print(error(f4, x, y))

fx4  = sp.linspace(0, x[-1], 1000)
plt.plot(fx4, f4(fx4), linewidth = 3)

plt.legend(["d=%i" % f4.order], loc = "upper left")

inflection = 3.5*24
xa = x[:inflection] #before the inflection
ya = y[:inflection]
xb = x[inflection:] #after
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xa, ya, 1))

fa_error = error(fa,xa, ya)
fb_error = error(fb, xb, yb)
print("Error %d", str(fa+fb_error))
#print("Error inflection=%f" % (  ))


plt.show()