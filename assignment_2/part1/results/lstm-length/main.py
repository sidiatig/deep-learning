import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

path = './'

len_to_file = {}
for f in listdir(path):
    if isfile(join(path, f)) and f.endswith('.npy'):
        length = int(f[f.find('len') + 3: f.find('.npy')])
        len_to_file[length] = f

lengths = sorted(len_to_file.keys())
accs = [np.load(len_to_file[i]) for i in lengths]

plt.figure(figsize=(4, 3))
plt.boxplot(accs, showfliers=False)
plt.xlabel('Length')
plt.ylabel('Accuracy')
plt.xticks([i+1 for i in range(len(lengths))], lengths)
plt.tight_layout()
plt.show()

