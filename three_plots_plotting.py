import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(6, 1.5))
x = np.arange(4,11)

# Policy C Done 2
ptc_done_modified = np.array([0,0,0.76,0.93,1,1,1])
ptc_done_abstracted = np.array([0,0.94,1,1,1,1,1])
ptc_done_both = np.array([0,0,0.76,0.93,1,1,1])


# Policy B Done 2
ptb_done_modified = np.array([0,0,0,0.378,0.5,0.62,0.62])
ptb_done_abstracted = np.array([0,0.19,0.12,0.5,0.55,0.62,0.62])
ptb_done_both = np.array([0,0,0,0.37,0.43,0.5,0.5])

# Policy B Empty
ptb_empty_modified = np.array([0,0,0.19,0.12,0.12,0.05,0.12])
ptb_empty_abstracted = np.array([0,0.11,0.24,0.05,0.12,0.12,0.12])
ptb_empty_both = np.array([0,0,0.19,0.12,0.19,0.06,0.06])

# Policy C Empty
ptc_empty_modified = np.array([0,0,0.06,0.06,0,0,0])
ptc_empty_abstracted = np.array([0,0,0,0,0,0,0])
ptc_empty_both = np.array([0,0,0.06,0.06,0,0,0])

# Modified MDPs
ax1.plot(x, ptc_done_modified, label='$\\mathcal{P}^{\\tau_C}(\\diamond done=2)$')
ax1.plot(x, ptb_done_modified, label='$\\mathcal{P}^{\\tau_B}(\\diamond done=2)$')
ax1.plot(x, ptb_empty_modified, label='$\\mathcal{P}^{\\tau_B}(\\diamond empty)$')
ax1.plot(x, ptc_empty_modified, label='$\\mathcal{P}^{\\tau_C}(\\diamond empty)$')
ax1.set_title('Modified MDPs')
ax1.set_xlabel('Maximal Fuel Level')
ax1.set_xlim(xmin=4,xmax=10)
ax1.set_xticks(x)
ax1.set_ylim(ymin=0)
ax1.set_yticks(np.arange(0,1.1,0.2))
ax1.grid(alpha=0.2)


#Abstracted Policies
ax2.plot(x, ptc_done_abstracted, label='$\\mathcal{P}^{\\tau_C}(\\diamond done=2)$')
ax2.plot(x, ptb_done_abstracted, label='$\\mathcal{P}^{\\tau_B}(\\diamond done=2)$')
ax2.plot(x, ptb_empty_abstracted, label='$\\mathcal{P}^{\\tau_B}(\\diamond empty)$')
ax2.plot(x, ptc_empty_abstracted, label='$\\mathcal{P}^{\\tau_C}(\\diamond empty)$')
ax2.set_xlim(xmin=4,xmax=10)
ax2.set_title('Abstracted $\pi$s')
ax2.set_xlabel('Abstracted Fuel Level')
ax2.set_xticks(x)
ax2.set_ylim(ymin=0)
ax2.set_yticks(np.arange(0,1.1,0.2))
ax2.grid(alpha=0.2)

#Both Policies
ax3.plot(x, ptc_done_modified, label='$\\mathcal{P}^{\\tau_C}(\\diamond done=2)$')
ax3.plot(x, ptb_done_both, label='$\\mathcal{P}^{\\tau_B}(\\diamond done=2)$')
ax3.plot(x, ptb_empty_both, label='$\\mathcal{P}^{\\tau_B}(\\diamond empty)$')
ax3.plot(x, ptc_empty_both, label='$\\mathcal{P}^{\\tau_C}(\\diamond empty)$')

ax3.set_title('Both')
ax3.set_xlabel('Maximal Fuel Level')
ax3.set_ylabel('Fuel Level')
ax3.set_xlim(xmin=4,xmax=10)
ax3.set_xticks(x)
ax3.set_ylim(ymin=0)
ax3.set_yticks(np.arange(0,1.1,0.2))
ax3.grid(alpha=0.2)
plt.legend(bbox_to_anchor=(1.15,1.2), loc="upper left")
plt.tight_layout()

plt.show()