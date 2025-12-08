#!/usr/bin/env python3


import matplotlib.pyplot as plt

targets = [[-167297.54344731, -167301.10372634, -167308.44207031, -167312.11313287],[-145434.59446349, -145430.40561915, -145432.73985156, -145432.6785802]]
preds = [[-167298.46656051, -167300.59898357, -167308.72554459, -167311.71904965],[-145433.43693344, -145429.41508183, -145432.73929695, -145431.94648615]]

#targets = [[-167297.542, -167301.103, -167308.448, -167312.116],[-145434.544, -145430.403, -145432.739, -145432.657]]
#preds = [[-167298.461, -167300.593, -167308.725, -167311.718],[-145433.435, -145429.411, -145432.732, -145431.947]]

#preds = [[-806.05867847, -806.12472625, -806.07960117, -806.06081532], [-525.55931051, -525.55932241, -525.5549579,  -525.55570061]]
#targets = [[-806.05791555, -806.12391602, -806.07879219, -806.06000536], [-525.55807966, -525.5598365,  -525.55437007, -525.55485379]]

# Create Figure
fig, axs = plt.subplots(nrows=1, ncols=2)

#for ax, name, target_es, pyscf_es in zip(axs, names2, target_Es, pyscf_Es):
for i in range(2):

    # Square plot    
    #min_max = [min(targets[i]+preds[i]), max(targets[i]+preds[i])]
    #offset = 2
    #axs[i].set_ylim(min_max[0]-offset, min_max[1]+offset)
    #axs[i].set_xlim(min_max[0]-offset, min_max[1]+offset)
    #axs[i].set_aspect("equal")

    #axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
    #axs[i].yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #from matplotlib.ticker import ScalarFormatter
    
    #fmt = ScalarFormatter(useOffset=True, useMathText=True)
    #fmt.set_powerlimits((-3, 4))
    #axs[i].xaxis.set_major_formatter(fmt)
    #axs[i].yaxis.set_major_formatter(fmt)
    #axs[i].ticklabel_format(style='sci', axis='x', useOffset=True)
    #axs[i].ticklabel_format(style='sci', axis='y', useOffset=True)
    
    # Draw the scatter plot and marginals.
    #y_min_max = [min(preds[i]), max(preds[i])]
    #xy1, xy2 = [y_min_max[0], y_min_max[0]], [y_min_max[1], y_min_max[1]]
    #axs[i].axline(xy1, xy2, linestyle=":", c='black', alpha=0.8)
    axs[i].scatter(targets[i], preds[i])

#fig.set_figheight(6)
#fig.set_figwidth(12)

# Save Plot
from datetime import datetime
time_date = datetime.now().strftime('%H%M_%d%m')
plt.savefig('./thesis/plot_all_parity/plots/all_parity2_{}.png'.format(time_date), bbox_inches='tight')

plt.show()
    
    
    
    
    
    
