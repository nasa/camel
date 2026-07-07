import plotly.graph_objects as go

# FIG LAYOUT
font_style = {
    'color' : '#2b2323'
}

margin_style = {
    'b': 10,
    'l': 50,
    'r': 8,
    't': 50,
    'pad': 0
}

xaxis_style = {
    'linewidth' : 1,
    'linecolor' : 'rgba(0, 0, 0, 0.35%)',
    'showgrid' : False,
    'zeroline' : False
}

yaxis_style = {
    'linewidth' : 1,
    'linecolor' : 'rgba(0, 0, 0, 0.35%)',
    'showgrid' : True,
    'gridwidth' : 1,
    'gridcolor' : 'rgba(0, 0, 0, 0.11%)',
    'zeroline' : False
}

my_figlayout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font = font_style,
    margin = margin_style,
    xaxis = xaxis_style,
    yaxis = yaxis_style,
    height = 300
)

# TRACES LAYOUT
obs_linelayout = {
    'width' : 2,
    'color' : 'gray'
}

my_tablelayout = {
    "style_cell": {
        "textAlign": "left",
        "padding": "10px",
        "backgroundColor": "rgba(0,0,0,0)",  # transparent
        "color": font_style,
        "border": "none",
        "fontSize": "16px",
    },
    "style_header": {
        "backgroundColor": "rgba(0,0,0,0.3)",
        "color": font_style,
        "border": "none",
        "borderBottom": "2px solid #d1d5db",  # header separator
    },
    "style_data": {
        "borderBottom": "1px solid #e5e7eb",  # row lines
    },
    "style_data_conditional": [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "rgba(0,0,0,0.1)",
        }
    ],
    "style_table": {"overflowX": "auto"},
}