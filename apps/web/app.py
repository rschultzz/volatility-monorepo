
# Minimal Dash entrypoint (replace with your actual app files)
import os
from dash import Dash, html
from flask import Flask

server = Flask(__name__)
app = Dash(__name__, server=server)

app.layout = html.Div([
    html.H3("Volatility Dash • Monorepo Skeleton"),
    html.P("Replace this with your existing Dash app code.")
])

if __name__ == "__main__":
    # Local dev server
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=True)
