import random

import networkx as nx
from networkx.algorithms import approximation

from pyecharts import options as opts
from pyecharts.charts import Graph, Bar, Line, Radar

import prettytable


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def preprocessing(path):
    with open(path, 'r') as book:
        lines = book.readlines()

    lines = [x.strip() for x in lines if x[0] not in ['%', '*']]
    id_name_pairs = [x.replace('"', '').split(' ', 1) for x in lines[:6927]]
    id_name_pairs = [x[1] for x in id_name_pairs]

    id_id_pairs = [x.split() for x in lines[6927:]]
    id_id_pairs = [[int(y) - 1 for y in records] for records in id_id_pairs]
    id_id_list = [None] * (id_id_pairs[-1][0] + 1)
    for x in id_id_pairs:
        id_id_list[x[0]] = x[1:] if id_id_list[x[0]] is None else id_id_list[x[0]] + x[1:]

    return id_name_pairs, id_id_list


def create_network():
    G = nx.Graph()

    for index, name in enumerate(id_name):
        G.add_node(index, name=name, E_numbers=1 if index < 507 else 2)

    G.nodes[6926]['E_numbers'] = 0

    for x, y in enumerate(id_id):
        for z in y:
            G.add_edge(x, z)

    return G


def draw_degree():
    degree = nx.degree(G)
    degree_sort = sorted(degree, key=lambda x: x[1], reverse=True)
    e0 = [opts.BarItem(name=G.nodes[degree_sort[0][0]], value=degree_sort[0][1])]
    e1 = [opts.BarItem(name=G.nodes[x[0]]['name'], value=x[1]) for x in degree_sort if G.nodes[x[0]]['E_numbers'] == 1][
         :50]
    e2 = [opts.BarItem(name=G.nodes[x[0]]['name'], value=x[1]) for x in degree_sort if G.nodes[x[0]]['E_numbers'] == 2][
         :50]

    xaxis = [x + 1 for x in range(50)]
    c = (
        Bar()
            .add_xaxis(xaxis)
            .add_yaxis("Erdos_number is 0", e0, category_gap=0, itemstyle_opts=opts.ItemStyleOpts(color='#d48265'),
                       gap="0%")
            .add_yaxis("Erdos_number is 1", e1, category_gap=0, itemstyle_opts=opts.ItemStyleOpts(color='#749f83'),
                       gap="0%")
            .add_yaxis("Erdos_number is 2", e2, category_gap=0, gap="0%")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False), axisline_opts=opts.AxisOpts(interval=1),
                             markline_opts=opts.MarkLineOpts(
                                 data=[
                                     opts.MarkLineItem(type_="average", name="Average"),
                                 ]
                             ), )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="Top 50 degree authors"),
            datazoom_opts=opts.DataZoomOpts(),
            yaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .render("bar_stack1.html")
    )


def draw_degree_histogram():
    degree_histogram = nx.degree_histogram(G)
    c = (
        Line()
            .add_xaxis([x for x in range(len(degree_histogram))])
            .add_yaxis("Number of authors", degree_histogram, is_smooth=True)
            .set_series_opts(
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="Maximum"),
                ]
            ),
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="Degree Histogram"),
            datazoom_opts=opts.DataZoomOpts(),
            xaxis_opts=opts.AxisOpts(
                name="Degree",
                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                is_scale=False,
                boundary_gap=False,
            ),
        )
            .render("Degree_Histogram.html")
    )


def draw_centrality():
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    degree_centrality_sort = sorted(degree_centrality, key=lambda x: degree_centrality[x], reverse=True)
    closeness_centrality_sort = sorted(closeness_centrality, key=lambda x: closeness_centrality[x], reverse=True)
    betweenness_centrality_sort = sorted(betweenness_centrality, key=lambda x: betweenness_centrality[x], reverse=True)
    h_authors = set.intersection(
        set(betweenness_centrality_sort[:15]),
        set(closeness_centrality_sort[:15]),
        set(degree_centrality_sort[:15]))

    data = [[G.nodes[x]["name"], [betweenness_centrality[x], closeness_centrality[x], degree_centrality[x]]] for x in
            h_authors]

    c = (
        Radar()
            .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Betweenness centrality [0,1]", max_=1, min_=0),
                opts.RadarIndicatorItem(name="Closeness centrality [0,0.6]", max_=0.6, min_=0),
                opts.RadarIndicatorItem(name="Degree centrality [0,0.1]", max_=0.1, min_=0),
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

    c.render("radar_angle_radius_axis.html")


def draw_the_whole_graph():
    nodes = [opts.GraphNode(
        name=G.nodes[x]['name'],
        value=G.degree[x],
        symbol_size=G.degree[x] / 10,
        category=G.nodes[x]['E_numbers']
    )
        for x in G.nodes]

    links = [opts.GraphLink(source=G.nodes[x]['name'], target=G.nodes[y]['name']) for x, y in G.edges]

    categories = [{'name': 'Erdos_number:' + str(x)} for x in range(3)]
    c = (
        Graph()
            .add(
            series_name="",
            nodes=nodes,
            links=links,
            layout='circular',
            is_roam=True,
            is_focusnode=True,
            label_opts=opts.LabelOpts(is_show=False),
            is_draggable=True,
            categories=categories,
            # repulsion=100
            # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7),
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph with \n authors degrees"))
    )
    c.render("Graph with authors degrees.html")


def draw_chain_decomposition():
    chain_decomposition = list(nx.chain_decomposition(G))
    longest_chain = sorted(chain_decomposition, key=lambda x: (len(x)), reverse=True)[0]
    nodes = [opts.GraphNode(name=G.nodes[x[0]]['name']) for x in longest_chain]
    nodes.append(opts.GraphNode(name=G.nodes[longest_chain[-1][1]]['name'], label_opts=opts.LabelOpts(color='#d48265')))
    nodes[0] = opts.GraphNode(name=G.nodes[longest_chain[0][0]]['name'], label_opts=opts.LabelOpts(color='#749f83'))
    links = [opts.GraphLink(source=G.nodes[x]['name'], target=G.nodes[y]['name']) for x, y in longest_chain]
    c = (
        Graph()
            .add(
            series_name="",
            nodes=nodes,
            links=links,
            layout='force',
            is_roam=True,
            is_focusnode=True,
            label_opts=opts.LabelOpts(is_show=False),
            is_draggable=True,
            repulsion=100,
            linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7),
        )
    )
    c.render("Graph with longest chain.html")


def draw_k_cores():
    k_cores = nx.k_core(G)
    nodes = [opts.GraphNode(name=k_cores.nodes[x]['name'], value=k_cores.degree[x], symbol_size=k_cores.degree[x]) for x
             in k_cores.nodes]
    links = [opts.GraphLink(source=k_cores.nodes[x]['name'], target=k_cores.nodes[y]['name']) for x, y in k_cores.edges]
    c = (
        Graph()
            .add(
            series_name="",
            nodes=nodes,
            links=links,
            layout='force',
            is_roam=True,
            is_focusnode=True,
            label_opts=opts.LabelOpts(is_show=False),
            is_draggable=True,
            repulsion=10000,
            # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7),
        )
    )
    c.render("k_cores_subgraph.html")


def analyze_clustering():
    average_clustering_coefficient = approximation.average_clustering(G)
    average_clustering = nx.average_clustering(G)
    average_shortest_path_length = nx.average_shortest_path_length(G)
    local_efficiency = nx.local_efficiency(G)
    global_efficiency = nx.global_efficiency(G)
    table = prettytable.PrettyTable(
        ['Average clustering', 'Average clustering coefficient', 'Average shortest path length', 'Local efficiency',
         'Global efficiency'])
    table.add_row([average_clustering, average_clustering_coefficient, average_shortest_path_length, local_efficiency,
                   global_efficiency])
    print(table)


def analyze_distance():
    center = nx.center(G)
    barycenter = nx.barycenter(G)
    diameter = nx.diameter(G)
    table = prettytable.PrettyTable(['center', 'barycenter', 'diameter'])
    table.add_row([center, barycenter, diameter])
    print(table)


def analyze_small_world():
    # Small-world
    sigma = nx.sigma(G)
    omega = nx.omega(G)
    table = prettytable.PrettyTable(['sigma', 'omega'])
    table.add_row([sigma, omega])
    print(table)


def ranking():
    voterank = nx.voterank(G)
    table = prettytable.PrettyTable(
        ['Rank', 'Author ID', 'Name', 'Final voting score', 'Final voting ability'])
    for index, aid in enumerate(voterank):
        table.add_row([index, aid, G.nodes[aid]['name'], G.nodes[aid]['voterank'][0], G.nodes[aid]['voterank'][1]])
    print(table)


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


id_name, id_id = preprocessing('12_Mathematicians_Network.txt')
G = create_network()

# draw
draw_the_whole_graph()
draw_degree()
draw_degree_histogram()
draw_chain_decomposition()
draw_k_cores()
# draw_centrality()
