import matplotlib.pyplot as plt
import networkx as nx

n=20
k=6
plt.figure(figsize=[16,8])
plt.subplot(1,2,1)
fully_connected=nx.complete_graph(n)
nx.draw_circular(fully_connected)
plt.subplot(1,2,2)
smallworld=nx.watts_strogatz_graph(n,k,p=0.2)
nx.draw_circular(smallworld)

plt.savefig("../figs/small_world_all_comparison.pdf", bbox_inches='tight')