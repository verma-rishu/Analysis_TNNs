from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
import networkx as nx
import matplotlib.pyplot as plt

loader = EnglandCovidDatasetLoader()
# Load the EnglandCovidDataset
dataset = loader.get_dataset(8)
snapshot_1 = dataset[52]

# Access the node features (x) of the first graph
node_features = snapshot_1.x
edges = snapshot_1.edge_index


weights = snapshot_1.edge_attr
weights = weights.tolist()
#print(weights)

traget = snapshot_1.y


# # # Convert the PyTorch Geometric data object to a NetworkX graph
G = nx.DiGraph()
print(G.is_directed())

src_arr = edges[0]
dst_arr = edges[1]
src_arr = src_arr.tolist()
dst_arr = dst_arr.tolist()

# basic add nodes
for i in range(0, 129):
    G.add_node(i)

norm_weight = [(float(i)/max(weights))+0.02 for i in weights]
merged_list = [(src_arr[i], dst_arr[i], norm_weight[i]) for i in range(0, len(dst_arr))]

G.add_weighted_edges_from(merged_list)
nodelist = G.nodes()
widths = nx.get_edge_attributes(G, 'weight')
print(widths)
pos = nx.shell_layout(G)

nx.draw_networkx_nodes(G,pos,
                       nodelist=nodelist,
                       node_size=200,
                       node_color='black',
                       alpha=0.7)
nx.draw_networkx_edges(G,pos,
                       edgelist = widths.keys(),
                       width=list(widths.values()),
                       edge_color='tab:red',
                       alpha=0.9)
nx.draw_networkx_labels(G, pos=pos,
                        font_size = 6,
                        labels=dict(zip(nodelist,nodelist)),
                        font_color='white')


plt.tight_layout()
#plt.show()
plt.savefig('snapshot_52.png',  dpi=1000)