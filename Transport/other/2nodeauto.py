import pandas as pd
import networkx as nx
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# 创建Dash应用
app = Dash(__name__)

# 应用布局
app.layout = html.Div([
    html.H1("网络连通性可视化"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'PEMS03', 'value': 'PEMS03.csv'},
            {'label': 'PEMS04', 'value': 'PEMS04.csv'},
            {'label': 'PEMS07', 'value': 'PEMS07.csv'},
            {'label': 'PEMS08', 'value': 'PEMS08.csv'}
        ],
        value='PEMS04.csv',
        clearable=False
    ),
    dcc.Graph(id='network-graph')
])

# 回调函数，当下拉菜单选择改变时触发
@app.callback(
    Output('network-graph', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_graph(selected_dataset):
    try:
        df = pd.read_csv(selected_dataset)
        G = nx.from_pandas_edgelist(df, 'from', 'to')

        # 为了更好的布局，我们调整spring_layout的k值
        pos = nx.spring_layout(G, k=0.5, iterations=20)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                line_width=2))

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'#{node}: {len(adjacencies[1])} connections')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig
    except Exception as e:
        return go.Figure()  # 在出现错误时返回一个空图表

# 运行服务器
if __name__ == '__main__':
    app.run_server(debug=True)
