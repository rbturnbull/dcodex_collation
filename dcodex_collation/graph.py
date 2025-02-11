from .models import Alignment
import networkx as nx
from lxml import etree


def build_subtree(graph, node, stack:list[etree.Element]):
    # Check if this node has been visited
    if hasattr(node, 'stack'):
        # If it has been visited, then stop traversing down this lineage
        return

    # Check if this node merges lineages
    ancestors = list(nx.ancestors(graph, node))
    if len(ancestors) >= 2:
        # Check to make sure that all ancestors have the same stack (if they have been visited before)
        for ancestor in ancestors:
            if hasattr(ancestor, 'stack'):
                assert ancestor.stack[:-1] == stack[:-1]
        
        # if there is a merger, then the stack is made smaller with FILO
        print('pop')
        parent = stack.pop()
    else:
        parent = stack[-1]
    

    # save this stack on the node
    # This also lets the function know that this node has been visited
    graph.nodes[node]["stack"] = stack

    # Add XML for this node
    for word in node.split("+"):
        etree.SubElement(parent, "w").text = word

    # Check if the node splits
    children = list(graph.successors(node))
    if len(children) == 0:
        # if we've hit the end, then we stop traversing completely
        return True

    # If there is only one child, then we don't need to create an <app> element
    if len(children) == 1:
        return build_subtree(graph, children[0], stack=stack)

    etree.SubElement(parent, "app")
    for child in children:
        rdg = etree.SubElement(parent, "rdg")
        build_subtree(graph, child, stack=stack + [rdg])


def convert_to_xml(graph, verse_ref:str="") -> etree.Element:
    if not nx.is_directed_acyclic_graph(graph):
        return None

    start_nodes = [n for n, d in graph.in_degree() if d == 0]
    if len(start_nodes) != 1:
        return None
    start = start_nodes[0]

    root = etree.Element("ab")
    if verse_ref:
        root.attrib['n'] = verse_ref

    if not build_subtree(graph, start, stack=[root]):
        return None

    return root


# def create_xml_node(node_text):
#     node = etree.Element("node")
#     node.text = node_text
#     return node


# def convert_to_xml_auto(graph):
#     if not nx.is_directed_acyclic_graph(graph):
#         return None

#     def build_subtree(parent, node):
#         children = list(graph.successors(node))
#         if len(children) == 0:
#             parent.append(create_xml_node(node))
#             return True

#         split = etree.Element("split")
#         for child in children:
#             option = etree.Element("option")
#             if not build_subtree(option, child):
#                 return False
#             split.append(option)
#         parent.append(create_xml_node(node))
#         parent.append(split)
#         return True

#     # Identify the start nodes (nodes with no predecessors)
#     start_nodes = [n for n, d in graph.in_degree() if d == 0]
#     if len(start_nodes) != 1:
#         return None

#     root = etree.Element("dag")
#     if not build_subtree(root, start_nodes[0]):
#         return None

#     return root


def variant_graph(alignment:Alignment):
    G = nx.DiGraph()
    start_node = "start"
    G.add_node(start_node, column_id=0, column=None)
    current_nodes = {}
    for row in alignment.row_set.all():
        current_nodes[row.transcription.manuscript.siglum] = start_node

    for column in alignment.column_set.all():
        print(column)
        for state, siglum in column.cell_set.values_list("state__text", "row__transcription__manuscript__siglum"):
            if not state or state.startswith("OMIT"):
                continue
            current_node = current_nodes[siglum]
            new_node = f'{column.order}-{state}'
            G.add_node(new_node, column_id=column.id, column=column)
            G.add_edge(current_node, new_node)
            mss = G.edges[current_node, new_node].get("mss", None)
            if mss:
                mss.add(siglum)
            else:
                mss = {siglum,}
            G.edges[current_node, new_node].update({"mss":mss})
            current_nodes[siglum] = new_node

    end_node = "end"
    G.add_node(end_node, column_id=-1, column=None)

    for row in alignment.row_set.all():
        current_node = current_nodes[row.transcription.manuscript.siglum]
        siglum = row.transcription.manuscript.siglum
        G.add_edge(current_node, end_node)
        mss = G.edges[current_node, end_node].get("mss", None)
        if mss:
            mss.add(siglum)
        else:
            mss = {siglum,}
        G.edges[current_node, end_node].update({"mss":mss})

    for start, end, data in G.edges(data=True):
        assert 'mss' in data

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
    # for node, data in G.nodes(data=True):
    #     data.clear()

    # result = convert_to_xml(G, verse_ref=alignment.verse.url_ref())
    # if result:
    #     print("can be represented as xml")
    #     # print(tostring(result))
    #     tree = etree.ElementTree(result)
    #     tree.write("variant-graph.xml",encoding="UTF-8", pretty_print=True)

    return G