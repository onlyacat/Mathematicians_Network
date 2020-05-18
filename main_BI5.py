import random

import pandas as pd
import networkx as nx
import prettytable
from pyecharts import options as opts
from pyecharts.charts import Graph, Bar, Line, Radar
from networkx.algorithms import approximation
from networkx.algorithms import community


def extract_data():
    edges = pd.read_excel('got_edges.xlsx')
    nodes = pd.read_excel('got_nodes.xlsx')

    G = nx.Graph()

    nodes.apply(lambda x: G.add_node(x.Id, name=x.Label), axis=1)
    edges.apply(lambda x: G.add_edge(x.Source, x.Target, weight=x.Weight), axis=1)

    return G


G = extract_data()


def analyze_basics():
    density = nx.density(G)
    average_degree = sum(dict(nx.degree(G)).values()) / len(dict(nx.degree(G)).values())  # print
    number_of_nodes = len(G.nodes)
    number_of_edges = len(G.edges)
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    table = prettytable.PrettyTable(
        ['Number of Nodes', 'Number Of Edges', 'Density', 'Average Degree', 'Degree Assortativity'])
    table.add_row([number_of_nodes, number_of_edges, density, average_degree, degree_assortativity])
    print(table)


analyze_basics()


def draw_the_whole_graph():
    nodes = [opts.GraphNode(
        name=x,
        value=G.degree[x],
        symbol_size=G.degree[x] / 10,
    )
        for x in G.nodes]

    links = [opts.GraphLink(source=x, target=y, value=G.edges[x, y]['weight'],linestyle_opts=opts.LineStyleOpts(width=G.edges[x, y]['weight'])
                            ) for x, y in G.edges]
    [math.ceil(G.edges[x, y]['weight'] / max_edge * 10) * 3 for x, y in G.edges]
    c = (
        Graph().add(
            series_name="",
            nodes=nodes,
            links=links,
            layout='force',
            is_roam=True,
            is_focusnode=True,
            label_opts=opts.LabelOpts(is_show=False),
            is_draggable=True,
            # repulsion=100
            # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7),
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph with \n authors degrees"))
    )
    return c


# c = draw_the_whole_graph()
# c.render("1.html")


def connected_detection():
    connected = nx.is_connected(G)
    GG = None
    if not connected:
        components = list(nx.connected_components(G))

        table = prettytable.PrettyTable(
            ['connected', 'Nodes Of Smaller Components', 'Smaller Components'])
        table.add_row([connected, min([len(x) for x in components]), components[1]])
        print(table)

        GG = G.subgraph(components[0])
        print(nx.info(GG))

    return GG


GG = connected_detection()


def analyze_clustering(G):
    average_clustering_coefficient = approximation.average_clustering(G)
    average_clustering = nx.average_clustering(G)
    average_shortest_path_length = nx.average_shortest_path_length(G)
    local_efficiency = nx.local_efficiency(G)
    global_efficiency = nx.global_efficiency(G)
    table = prettytable.PrettyTable(
        ['Average clustering', 'Average clustering coefficient', 'Average shortest path length'])
    table.add_row([average_clustering, average_clustering_coefficient, average_shortest_path_length])
    print(table)
    table = prettytable.PrettyTable(['Local efficiency', 'Global efficiency'])
    table.add_row([local_efficiency, global_efficiency])
    print(table)


# analyze_clustering(GG)


def analyze_distance(G):
    center = nx.center(G)
    barycenter = nx.barycenter(G)
    diameter = nx.diameter(G)
    table = prettytable.PrettyTable(['center', 'barycenter', 'diameter'])
    table.add_row([center, barycenter, diameter])
    print(table)


# analyze_distance(GG)


def analyze_small_world(G):
    # Small-world
    sigma = nx.sigma(G)
    omega = nx.omega(G)
    table = prettytable.PrettyTable(['sigma', 'omega'])
    table.add_row([sigma, omega])
    print(table)


# analyze_small_world(GG)


# def ranking(G):
#     voterank = nx.voterank(G)
#     table = prettytable.PrettyTable(
#         ['Rank', 'Author ID', 'Name', 'Final voting score', 'Final voting ability'])
#     for index, aid in enumerate(voterank[:15]):
#         table.add_row([index, aid, G.nodes[aid]['name'], G.nodes[aid]['voterank'][0], G.nodes[aid]['voterank'][1]])
#     print(table)
#
# ranking(GG)


def calculate_community():
    cate = list(community.asyn_lpa_communities(G, 'weight'))



def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def draw_centrality():
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    degree_centrality_sort = sorted(degree_centrality, key=lambda x: degree_centrality[x], reverse=True)
    closeness_centrality_sort = sorted(closeness_centrality, key=lambda x: closeness_centrality[x], reverse=True)
    betweenness_centrality_sort = sorted(betweenness_centrality, key=lambda x: betweenness_centrality[x], reverse=True)
    top_authors = sorted(
        [(x, degree_centrality[x] + closeness_centrality[x] + betweenness_centrality[x]) for x in set.intersection(
            set(betweenness_centrality_sort[:30]), set(closeness_centrality_sort[:30]),
            set(degree_centrality_sort[:30]))], key=lambda x: x[1], reverse=True)

    table = prettytable.PrettyTable(
        ['Rank', 'Name', 'Degree centrality', 'Closeness centrality', 'Betweenness centrality', 'Sum centrality'])

    [table.add_row(
        [index + 1, value[0], round(degree_centrality[value[0]], 3), round(closeness_centrality[value[0]], 3),
         round(betweenness_centrality[value[0]], 3), round(value[1], 3)]) for index, value in enumerate(top_authors)]

    print(table)

    h_authors = set.intersection(
        set(betweenness_centrality_sort[:15]),
        set(closeness_centrality_sort[:15]),
        set(degree_centrality_sort[:15]))

    data = [[G.nodes[x]["name"], [betweenness_centrality[x], closeness_centrality[x], degree_centrality[x]]] for x in
            h_authors[:10]]

    c = (
        Radar()
            .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Betweenness centrality [0,0.15]", max_=0.15, min_=0),
                opts.RadarIndicatorItem(name="Closeness centrality [0,0.6]", max_=0.6, min_=0),
                opts.RadarIndicatorItem(name="Degree centrality [0,0.35]", max_=0.35, min_=0),
            ],
            shape="circle",
            center=["50%", "50%"],
            radius="80%",
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            textstyle_opts=opts.TextStyleOpts(color="#000"),
        ).set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )

    for x in data:
        color = randomcolor()
        c.add(
            series_name=x[0],
            data=[x[1]],
            areastyle_opts=opts.AreaStyleOpts(opacity=0.1, color=color),
            linestyle_opts=opts.LineStyleOpts(width=1, color=color),
            label_opts=opts.LabelOpts(is_show=False)
        )

    return c


c = draw_centrality()
c.render("11.html")
