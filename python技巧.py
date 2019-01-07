import scipy.misc
import os
import numpy as np
os.chdir('D:\soft\git\kai\python_za')
aa = np.random.rand(100,151)
print(aa)
scipy.misc.toimage(aa,cmin = 0 , cmax = 1).save('random.jpg')
scipy.misc.imsave('random1.jpg',aa)
aa = np.array([0,0,1])
np.argmax(aa)