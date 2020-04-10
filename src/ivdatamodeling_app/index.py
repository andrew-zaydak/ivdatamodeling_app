import dash_html_components as html

from .app import app
from .utils import DashRouter, DashNavBar
from .pages import page2, page3, create_model
from .components import fa


# Ordered iterable of routes: tuples of (route, layout), where 'route' is a
# string corresponding to path of the route (will be prefixed with Dash's
# 'routes_pathname_prefix' and 'layout' is a Dash Component.
urls = (
    ("", create_model.get_layout),
    ("create-model", create_model.get_layout),
    ("page2", page2.layout),
    ("page3", page3.layout),
)

# Ordered iterable of navbar items: tuples of `(route, display)`, where `route`
# is a string corresponding to path of the route (will be prefixed with
# 'routes_pathname_prefix') and 'display' is a valid value for the `children`
# keyword argument for a Dash component (ie a Dash Component or a string).
nav_items = (
    ("create-model", html.Div([fa("fas fa-rocket"), "Create Model"])),
    ("page2", html.Div([fa("fas fa-chart-area"), "Models"])),
    ("page3", html.Div([fa("fas fa-chart-line"), "Data"])),
)
#("character-counter", html.Div([fa("fas fa-rocket"), "Create Model"])),

router = DashRouter(app, urls)
navbar = DashNavBar(app, nav_items)
