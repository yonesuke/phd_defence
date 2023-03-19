import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

num = 10
G = nx.generators.classic.circulant_graph(num, [1,2,3,5,7,8,9])

plt.figure(figsize=[8,8])
nx.draw_circular(G, with_labels=True, node_color="lightgreen")
plt.axis("off")
plt.savefig("../figs/circulant_net.pdf", bbox_inches="tight", transparent=True)
