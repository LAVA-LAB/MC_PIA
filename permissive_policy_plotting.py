from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 2.3))
plt.rcParams['text.usetex'] = True

plt.xlabel("Permissive Policy Starting Range")
plt.ylabel("Probability")

# Pmin/Pmax done2
x = np.arange(4,11)
ptc_max = np.array([])
ptc_min = np.array([])
plt.plot(x,ptc_max,color="violet", label='$\\mathcal{P}^{\\tau_C}_{max}(\\diamond done=2)$')
plt.plot(x,ptc_min,'--',color="violet", label='$\\mathcal{P}^{\\tau_C}_{min}(\\diamond done=2)$')

# Pmin/Pmax empty
x = np.arange(4,11)
empty_ptb_max = np.array([])
empty_ptb_min = np.array([])
plt.plot(x,empty_ptb_max,color="red", label='$\\mathcal{P}^{\\tau_B}_{max}(\\diamond empty)$')
plt.plot(x,empty_ptb_min,'--',color="red", label='$\\mathcal{P}^{\\tau_B}_{min}(\\diamond empty)$')


# Pmin/Pmax done2
x = np.arange(4,11)
ptb_max = np.array([])
ptb_min = np.array([])
plt.plot(x,ptb_max,color="blue", label='$\\mathcal{P}^{\\tau_B}_{max}(\\diamond done=2)$')
plt.plot(x,ptb_min,'--',color="blue", label='$\\mathcal{P}^{\\tau_B}_{min}(\\diamond done=2)$')


# Pmin/Pmax empty
empty_ptc_max = np.array([])
plt.plot(x,empty_ptc_max,color="cyan", label='$\\mathcal{P}^{\\tau_C}_{max}(\\diamond empty)$')
empty_ptc_min = np.array([])
plt.plot(x,empty_ptc_min,'x-',color="cyan", label='$\\mathcal{P}^{\\tau_C}_{min}(\\diamond empty)$')

plt.xlim(xmin=4,xmax=10)
plt.xticks(x)
plt.ylim(ymin=0)
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(bbox_to_anchor=(1.03,1.05), loc="upper left")
plt.tight_layout()
plt.grid(alpha=0.2)
plt.show()
