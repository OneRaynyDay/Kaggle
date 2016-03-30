import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig,(ax,ax2,ax3) = plt.subplots(1, 3, sharey=False)

# plot the same on all the subplots
ax.plot(x,y1,'r-', x,y2,'b-')
ax2.plot(x,y1,'r-', x,y2,'b-')
ax3.plot(x,y1,'r-', x,y2,'b-')

# now zoom in each of the subplots 
ax.set_xlim([ x[0],x[1]])
ax2.set_xlim([ x[1],x[2]])
ax3.set_xlim([ x[2],x[3]])

# set the x axis ticks 
for axx,xx in zip([ax,ax2,ax3],x[:-1]):
  axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
ax3.xaxis.set_major_locator(ticker.FixedLocator([x[-2],x[-1]]))  # the last one

# EDIT: add the labels to the rightmost spine
for tick in ax3.yaxis.get_major_ticks():
  tick.label2On=True

# stack the subplots together
plt.subplots_adjust(wspace=0)

plt.show()
