# desmos_in_manim

A Python library to integrate [Desmos](https://www.desmos.com/) graphs into [Manim](https://www.manim.community/) animations.

## Features

- Render Desmos 2D and 3D graphs as Manim objects
- Animate Desmos parameters, actions, and camera movements
- Support for graph translation, zoom, and 3D rotation animations

## Installation

### From GitHub

```bash
pip install git+https://github.com/physy/desmos_in_manim.git
playwright install chromium
```

with uv:

```bash
uv add git+https://github.com/physy/desmos_in_manim.git
uv run playwright install chromium
```

## Requirements

- Python 3.10+
- Manim 0.19.0+

## Performance Tips

Since each frame requires a screenshot from the Desmos browser instance, rendering can be slow. During development, it's recommended to lower the frame rate:

```python
# manim.cfg
[CLI]
frame_rate = 10
```

## Quick Start

```python
from manim import *
from desmos_in_manim import DesmosGraph

# Load Desmos state from a JSON file (exported from Desmos)
# You can get the state by executing Calc.getState() in the console of your browser
with open("graph.json", "r") as f:
    state = f.read()

# Create the graph object outside the Scene for reuse
graph = DesmosGraph(
    state=state, # State should be a string of json
    width=1920,
    height=1080,
    background_color="#000000",
    graph_settings={"graphLineWidth": 4},
)

class MyScene(Scene):
    def construct(self):
        self.add(graph)
        self.wait(2)
```

## Usage

### Creating a Graph

```python
graph = DesmosGraph(
    state=state,                      # Desmos state JSON string
    width=1920,                       # Screenshot width
    height=1080,                      # Screenshot height
    background_color="#000000",       # Background color (hex)
    is3D=False,                       # Set True for 3D graphs
    graph_settings={                  # Optional graph styling
        "graphLineWidth": 4,
    },
)
```

### Setting View Bounds

```python
graph.set_mathBounds({
    "xmin": -10, "xmax": 10,
    "ymin": -10, "ymax": 10
})
```

### Parameter Animation

Animate slider variables defined in your Desmos graph. You can use the expression ID or the variable name.

```python
# Animate parameter by expression ID
self.play(graph.animate_parameter("4", 0, 1, run_time=2, rate_func=smooth))

# Animate multiple parameters simultaneously
self.play(
    graph.animate_parameter("4", 0, 1, run_time=2, rate_func=smooth),
    graph.animate_parameter("35", 0, 1, run_time=2, rate_func=smooth),
    graph.animate_parameter("49", 0, 1, run_time=2, rate_func=smooth),
)

# Animate a named variable (LaTeX)
self.play(graph.animate_parameter("t_{1}", 0, 0.15, run_time=1, rate_func=smooth))
```

### Camera Animations

```python
# Translate the view
self.play(graph.animate_translation(dx=5, dy=3))

# Animate to specific bounds
self.play(graph.animate_bounds_transition(
    {"xmin": -2, "xmax": 5, "ymin": 0, "ymax": 4}
))

# Zoom in on a point
self.play(graph.animate_zoom(center_x=0, center_y=0, scale=2))
```

### Combining with Manim Objects

```python
class Example(Scene):
    def construct(self):
        self.add(graph)

        # Create Manim objects on top of Desmos graph
        arrow = Arrow(
            RIGHT,
            RIGHT+ UP * 2,
            color=BLUE_D,
            stroke_width=10,
        )
        label = Text("Force", font_size=40, color=BLUE_C)
        label.next_to(arrow, UP)

        self.play(Create(arrow), Write(label))
        self.wait(1)

        # Animate Desmos and Manim objects together
        self.play(
            graph.animate_parameter("87", 0, 0.15, run_time=1),
            FadeOut(arrow),
            FadeOut(label),
        )
```

### Custom Arrow Tips

The package includes custom arrow tips:

```python
from desmos_in_manim import ArrowSharpTriangleFilledTip

arrow = Arrow(
    start, end,
    tip_shape=ArrowSharpTriangleFilledTip,
    stroke_width=10,
)
```

### 3D Graphs

```python
graph_3d = DesmosGraph(
    state=state,
    is3D=True,
    calc_options={"perspectiveDistortion": 7},
)

# Set initial rotation
graph_3d.set_rotation_to(z_tip=PI/4, xy_rot=PI/2)

# Rotate the 3D view (relative)
self.play(graph_3d.animate_rotation(z_tip_delta=PI/4, xy_rot_delta=PI/2))

# Rotate to absolute angle
self.play(graph_3d.animate_rotation_to(z_tip=PI/3, xy_rot=PI))
```

### Using Updaters for Animation

Use Manim's updater system to continuously update Desmos parameters (e.g., for time-based animations):

```python
class WaveAnimation(Scene):
    def construct(self):
        graph = DesmosGraph(state="...", is3D=True)
        self.add(graph)

        # Track time
        self.t = 0

        # Update time parameter every frame
        def update_time(mob, dt):
            self.t += dt
            graph.set_expression({"id": "8", "latex": f"t={self.t:.2f}"})

        graph.add_updater(update_time)
        self.wait(5)  # Animation runs for 5 seconds

        # You can also add rotation updater for continuous rotation
        def rotate(mob, dt):
            graph.set_rotation_by(xy_rot_delta=-dt * PI / 12, update_display=False)

        graph.add_updater(rotate)
        self.wait(3)

        # Remove updater when done
        graph.remove_updater(rotate)
```

### Combining Updaters with Animations

```python
class CombinedAnimation(Scene):
    def construct(self):
        graph = DesmosGraph(state="...", is3D=True)
        graph.set_rotation_to(z_tip=PI/4, xy_rot=PI/2)
        self.add(graph)

        self.t = 0

        def update_time(mob, dt):
            self.t += dt
            graph.set_expression({"id": "8", "latex": f"t={self.t:.2f}"})

        graph.add_updater(update_time)

        # Run parameter animation while time updater is active
        self.play(
            graph.animate_parameter("29", -10, 15, run_time=4, rate_func=linear)
        )

        # Add rotation updater and play another animation
        def rotate(mob, dt):
            graph.set_rotation_by(xy_rot_delta=-dt * PI / 12, update_display=False)

        graph.add_updater(rotate)

        self.play(
            graph.animate_parameter("5", 0.8, 0.4, run_time=1, rate_func=smooth)
        )

        graph.remove_updater(rotate)
```

### Executing JavaScript

For advanced control, execute JavaScript directly on the Desmos calculator:

```python
graph.execute_js("Calc.setExpression({id: 'test', latex: 'y=2x'})")
```

## API Reference

### DesmosGraph

#### Constructor

| Parameter          | Type          | Default     | Description                                                        |
| ------------------ | ------------- | ----------- | ------------------------------------------------------------------ |
| `state`            | `str \| None` | `None`      | Desmos state JSON string                                           |
| `width`            | `int`         | `1920`      | Screenshot width                                                   |
| `height`           | `int`         | `1080`      | Screenshot height                                                  |
| `background_color` | `str \| None` | `"#000000"` | Background color (hex)                                             |
| `is3D`             | `bool`        | `False`     | Enable 3D graph mode                                               |
| `calc_options`     | `dict`        | `{}`        | Desmos calculator options (`Calc.controller.graphSettings.config`) |
| `graph_settings`   | `dict`        | `{}`        | Graph styling options (`Calc.controller.graphSettings`)            |
| `use_cache`        | `bool`        | `False`     | Cache screenshots                                                  |

#### Methods

| Method                   | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| `set_state(state)`       | Set Desmos state from JSON string                             |
| `get_state()`            | Get current Desmos state as dict                              |
| `set_expression(expr)`   | Set a single expression (e.g., `{"id": "1", "latex": "y=x"}`) |
| `set_expressions(exprs)` | Set multiple expressions                                      |
| `get_expressions()`      | Get all expressions                                           |
| `set_mathBounds(bounds)` | Set view bounds (`xmin`, `xmax`, `ymin`, `ymax`)              |
| `get_mathBounds()`       | Get current view bounds                                       |
| `set_parameter()`        | Set parameter value by name or ID                             |
| `set_blank()`            | Clear all expressions                                         |
| `update_display()`       | Manually refresh the graph image                              |
| `execute_js(script)`     | Execute JavaScript on Desmos calculator                       |
| `add_updater(func)`      | Add an updater function                                       |
| `remove_updater(func)`   | Remove an updater function                                    |
| `cleanup()`              | Close browser and clean up resources                          |

#### Action Methods

| Method                             | Description                            |
| ---------------------------------- | -------------------------------------- |
| `action_single_step(exp_id)`       | Execute a Desmos action once           |
| `action_multi_step(exp_id, steps)` | Execute a Desmos action multiple times |

#### 3D-only Methods

| Method                                       | Description                                           |
| -------------------------------------------- | ----------------------------------------------------- |
| `set_rotation_to(z_tip, xy_rot)`             | Set 3D rotation to absolute angles (radians)          |
| `set_rotation_by(z_tip_delta, xy_rot_delta)` | Rotate 3D view by relative angles                     |
| `get_current_orientation()`                  | Get current rotation as `{"zTip": ..., "xyRot": ...}` |

#### Animation Methods

| Method                                                  | Description                    |
| ------------------------------------------------------- | ------------------------------ |
| `animate_parameter(name, start, end, **kwargs)`         | Animate a slider parameter     |
| `animate_translation(dx, dy, dz, **kwargs)`             | Animate view translation       |
| `animate_bounds_transition(target_bounds, **kwargs)`    | Animate to target bounds       |
| `animate_zoom(center_x, center_y, scale, **kwargs)`     | Animate zoom                   |
| `animate_rotation(z_tip_delta, xy_rot_delta, **kwargs)` | Animate 3D rotation (relative) |
| `animate_rotation_to(z_tip, xy_rot, **kwargs)`          | Animate 3D rotation (absolute) |

### Arrow Tips

| Class                         | Description                              |
| ----------------------------- | ---------------------------------------- |
| `ArrowSharpTriangleTip`       | Sharp triangular arrow tip (stroke only) |
| `ArrowSharpTriangleFilledTip` | Sharp triangular arrow tip (filled)      |

## License

MIT
