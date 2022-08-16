from .models import Alignment
import networkx as nx

def variant_graph(alignment:Alignment):
    G = nx.DiGraph()
    start_node = "start"
    current_nodes = {}
    for row in alignment.row_set.all():
        current_nodes[row.transcription.manuscript.siglum] = start_node

    for column in alignment.column_set.all():
        print(column)
        for state, siglum in column.cell_set.values_list("state__text", "row__transcription__manuscript__siglum"):
            if not state:
                continue
            current_node = current_nodes[siglum]
            new_node = f'{column.order}-{state}'
            G.add_edge(current_node, new_node)
            current_nodes[siglum] = new_node

    end_node = "end"
    for row in alignment.row_set.all():
        current_node = current_nodes[row.transcription.manuscript.siglum]
        G.add_edge(current_node, end_node)



    # simplify
    edges_to_contract = []

    for node in G:
        if node == "start":
            continue
        out_edges = list(G.out_edges(node))
        print(node, len(out_edges), out_edges)
        if len(out_edges) == 1:
            out_node = out_edges[0][1]
            if node == "end":
                continue

            in_edges = G.in_edges(out_node)
            if len(in_edges) == 1:
                print(f"combine {node} and {out_node}")
                edges_to_contract.append( (node, out_node) )

    dropped_nodes = {}
    print(edges_to_contract)
    for edge in edges_to_contract:
        start = edge[0]
        end = edge[1]
        
        if start in dropped_nodes:
            start = dropped_nodes[start]

        new_name = f"{start}+{end}"
        nx.identified_nodes(G, start, end, self_loops=False, copy=False)
        print("rename", {start: new_name})
        print("b4", G.nodes())
        nx.relabel_nodes(G, {start: new_name}, copy=False)
        print("after", G.nodes())

        dropped_nodes[ edge[0] ] = new_name
        dropped_nodes[ edge[1] ] = new_name
        dropped_nodes[ start ] = new_name
        dropped_nodes[ end ] = new_name
        # break

    # Remove attributes
    for node, data in G.nodes(data=True):
        data.clear()

    return G