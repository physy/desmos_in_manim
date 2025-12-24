import asyncio
import tempfile
import os
import base64
import atexit
import math
import io
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Callable
from playwright.async_api import async_playwright
from manim import *
import nest_asyncio
import numpy as np
from PIL import Image
import re

nest_asyncio.apply()


class DesmosGraph(Group):
    """Display Desmos graph in Manim"""

    def __init__(
        self,
        state: str | None = None,
        width=1920,
        height=1080,
        background_color: str | None = "#000000",
        is3D: bool = False,
        calc_options: Dict = {},
        graph_settings: Dict = {},
        img_format: str = "png",
        use_cache: bool = False,
        **kwargs,
    ):
        """
        parameters:
        ----------
        state: str | None
            Desmos state (JSON string).
        width: int
        height: int
        is3D: bool = False
        calc_options: {
            colors: {string: string},
            fontSize: int,
            invertedColors: bool,
            beta3d: bool,
            advancedStyling: bool,
            authorMode: bool,
            disableLighting: bool,
            translucentOpacity: bool,
            backgroundColor: string,
            perspectiveDistortion: int,
        }
        graph_settings: {
            axisLineWidth: int = 1.5,
            axisLineOffset: int = 0.25,
            axisOpacity: int = 0.9,
            curveOpacity: int = 0.7,
            disableFill: bool = False,
            graphLineWidth: int = 2.5,
            highlight: bool = False,
            labelHangingColor: str = "rgba(150,150,150,1)",
            labelSize: int = 30,
            lastChangedAxis: str = "x",
            majorAxisOpacity: int = 0.4,
            minorAxisOpacity: int = 0.12,
            pixelsPerLabel: int = 80,
            pointLineWidth: int = 9,
            squareAxes: bool = False,
            shoeBox3D: bool = True,
        }
        img_format: str = "png"
            Image format to retrieve. 'png' or 'svg'. 3D doesn't support 'svg'.
        use_cache: bool = False
            Whether to cache images. If False, processes like get_state() can be skipped, potentially making it faster.
            Even if True, state calculation is still performed; only the screenshot process is skipped.
        """
        super().__init__(**kwargs)

        # Cache settings
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir = Path("media/desmos")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Current image
        self.current_image = None
        self.graph_width = width
        self.graph_height = height
        self.background_color = background_color
        self.is3D = is3D
        self.calc_options = calc_options
        # For 3D, backgroundColor is applied if set in calc_options
        # For 2D, need to fill with inequality
        if background_color is not None:
            if self.calc_options is None:
                self.calc_options = {}
            self.calc_options["backgroundColor"] = background_color
        self.graph_settings = graph_settings
        self.img_format = img_format if not is3D else "png"
        # TODO: svg doesn't display properly, so disable for now
        self.img_format = "png"
        # Set isPlaying to false
        asyncio.run(
            self._init_graph(
                re.sub(r".isPlaying.: *true", '"isPlaying": false', state or "")
            )
        )
        self.update_display()
        atexit.register(self.cleanup)

    async def _init_graph(self, state: str):
        self.playwright = await async_playwright().start()
        # 3D requires GPU enabled or it will be slow
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--enable-gpu"],
        )
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 600, "height": 400})
        # HTML longer than this may not be passable via data URL
        await self.page.goto(
            f"""
                data:text/html,
                <html>
                    <body>
                        <script src="https://www.desmos.com/api/v1.12/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
                        <div id="calc"></div>
                        <script>
                            window.Calc = Desmos.{ 'Calculator3D' if self.is3D else 'GraphingCalculator' }(document.getElementById("calc"));
                        </script>
                    </body>
                </html>
            """.strip()
        )
        await self.page.wait_for_function(
            "typeof window.Calc !== 'undefined'", timeout=15000
        )
        if state is not None:
            await self.page.evaluate(
                f"""
                    window.Calc.setState({state});
                """
            )
        if self.background_color is not None and self.is3D == False:
            await self.page.evaluate(
                f"""
                    Calc.controller.dispatch({{
                        type: "insert-item-at-index",
                        state: {{
                            id: Calc.controller.generateId(),
                            type: "expression",
                            latex: "1^{{x}}>0",
                            fillOpacity: "1",
                            color: "{self.background_color}"
                        }},
                        index: 0
                    }})
                """
            )
        if self.calc_options is not None:
            for key, value in self.calc_options.items():
                if isinstance(value, str):
                    value = f'"{value}"'
                elif isinstance(value, bool):
                    value = str(value).lower()
                await self.page.evaluate(
                    f"""
                        Calc.controller.graphSettings.config.{key} = {value};
                    """
                )
        if self.graph_settings is not None:
            for key, value in self.graph_settings.items():
                if isinstance(value, str):
                    value = f'"{value}"'
                elif isinstance(value, bool):
                    value = str(value).lower()
                await self.page.evaluate(
                    f"""
                        Calc.controller.graphSettings.{key} = {value};
                    """
                )

    def _generate_cache_key(self) -> str:
        """Generate cache key from current Desmos state"""
        try:
            # Get current state
            state = self.get_state()

            # Combine elements that affect cache key
            cache_data = {
                "state": state,
                "width": self.graph_width,
                "height": self.graph_height,
                "is3D": self.is3D,
                "img_format": self.img_format,
                "calc_options": self.calc_options,
                "graph_settings": self.graph_settings,
            }

            # Convert to JSON string and hash
            cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(cache_str.encode("utf-8")).hexdigest()[
                :16
            ]  # Shorten to 16 characters
        except Exception:
            # Return timestamp-based key on error
            import time

            return f"fallback_{int(time.time())}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        # If self.cache_dir is None, create cache directory in temp directory as fallback
        if self.cache_dir is None:
            # tempfile is already imported at the top of the file
            fallback_dir = Path(tempfile.gettempdir()) / "desmos"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = fallback_dir
        else:
            cache_dir = self.cache_dir

        extension = "svg" if self.img_format == "svg" else "png"
        return cache_dir / f"{cache_key}.{extension}"

    def _load_from_cache(self, cache_key: str):
        """Return file path if cache file exists"""
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            return str(cache_path)  # Return file path as string

        return None

    def _save_to_cache(self, cache_key: str, data, is_data_url: bool = False):
        """Save data to cache file"""
        cache_path = self._get_cache_path(cache_key)

        try:
            if self.img_format == "svg":
                # Save SVG string to file
                cache_path.write_text(data, encoding="utf-8")
            else:
                # For data URL, decode base64 and save
                if is_data_url:
                    header, base64_data = data.split(",", 1)
                    screenshot_bytes = base64.b64decode(base64_data)
                    cache_path.write_bytes(screenshot_bytes)
                else:
                    # Save NumPy array as PNG image
                    pil_image = Image.fromarray(data)
                    pil_image.save(cache_path, "PNG")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear cache directory"""
        try:
            if self.cache_dir is None:
                print("Cache directory is not set.")
                return
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
            print(f"Cache cleared: {self.cache_dir}")
        except Exception as e:
            print(f"Failed to clear cache: {e}")

    def get_cache_info(self):
        """Display cache information"""
        if self.cache_dir is None:
            print("Cache directory is not set.")
            return

        cache_files = list(self.cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [f.name for f in cache_files],
        }

    def execute_js(self, script: str, update_display: bool = True):
        """Execute JavaScript code"""
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.page.evaluate(script))
        if update_display:
            self.update_display()
        return res

    def update_calc_options(self, options: Dict):
        """Update Desmos calc_options"""
        for key, value in options.items():
            self.execute_js(
                f"""
                    Calc.controller.graphSettings.config.{key} = {str(value).lower() if isinstance(value, bool) else value};
                """
            )

    def update_graph_settings(self, settings: Dict):
        """Update Desmos graph_settings"""
        for key, value in settings.items():
            self.execute_js(
                f"""
                    Calc.controller.graphSettings.{key} = {str(value).lower() if isinstance(value, bool) else value};
                """
            )

    def _create_screenshot(self):
        """Take screenshot and return as file path or data"""
        if self.use_cache:
            return self._create_screenshot_with_cache()
        else:
            return self._create_screenshot_direct()

    def _create_screenshot_with_cache(self):
        """Take screenshot with caching"""
        # Generate cache key from state
        cache_key = self._generate_cache_key()

        # Try to load from cache
        cached_path = self._load_from_cache(cache_key)
        if cached_path is not None:
            return cached_path

        # If no cache, take new screenshot
        data = self._get_screenshot_data()

        # Save to cache
        if self.img_format == "svg":
            self._save_to_cache(cache_key, data)
        else:
            self._save_to_cache(cache_key, data, is_data_url=True)

        # Return saved file path
        return str(self._get_cache_path(cache_key))

    def _create_screenshot_direct(self):
        """Take screenshot directly without cache"""
        # Take screenshot directly and return data
        return self._get_screenshot_data()

    def _get_screenshot_data(self):
        """Get screenshot data (common process)"""
        if self.is3D:
            # Use square for 3D
            width = min(self.graph_width, self.graph_height) / 2
            return self.execute_js(
                f"""
                    screenshot = new Promise((resolve) => {{
                        Calc.controller.evaluator.notifyWhenSynced(() => {{
                            resolve(Calc.screenshot(
                                {{ width: {width} }}
                            ));
                        }});
                    }});
                """,
                update_display=False,
            )
        else:
            return self.execute_js(
                f"""
                    screenshot = new Promise((resolve) => {{
                        Calc.controller.evaluator.notifyWhenSynced(() => {{
                            Calc.controller.getGrapher().asyncScreenshot(
                                {{ width: {self.graph_width}, height: {self.graph_height}, showLabels: true, format: '{self.img_format}' }},
                                (url) => resolve(url)
                            );
                        }});
                    }});
                """,
                update_display=False,
            )

    def set_blank(self):
        """Set Desmos to blank"""
        self.execute_js(
            """
                window.Calc.setBlank();
            """
        )

    def set_state(self, state: str, update_display: bool = True):
        """Set Desmos state"""
        self.execute_js(
            f"""
                window.Calc.setState({state});
            """,
            update_display=update_display,
        )

    def get_state(self) -> Dict:
        """Get Desmos state"""
        return self.execute_js(
            """
                new Promise((resolve) => {
                    window.Calc.controller.evaluator.notifyWhenSynced(() => {
                        resolve(window.Calc.getState());
                    });
                });
            """,
            update_display=False,
        )

    def set_expression(self, expression: Dict, update_display: bool = True):
        """Set Desmos expression"""
        self.execute_js(
            f"""
                window.Calc.setExpression(
                    {expression}
                );
            """,
            update_display=update_display,
        )

    def set_expressions(self, expressions: List[Dict], update_display: bool = True):
        """Set Desmos expression list"""
        self.execute_js(
            f"""
                window.Calc.setExpressions(
                    {expressions}
                );
            """,
            update_display=update_display,
        )

    def get_expressions(self) -> List[Dict]:
        """Get Desmos expression list"""
        return self.execute_js(
            """
                window.Calc.getExpressions();
            """,
            update_display=False,
        )

    def set_mathBounds(self, bounds: Dict, update_display: bool = True):
        """Set Desmos display range"""
        bounds = dict(bounds)  # Create a copy to avoid modifying the original
        if (
            "xmin" in bounds
            and "xmax" in bounds
            and ("ymin" not in bounds or "ymax" not in bounds)
        ):
            # If y range is not specified, calculate it according to aspect ratio
            current_bounds = self.get_mathBounds()
            x_range = bounds.get("xmax", current_bounds["xmax"]) - bounds.get(
                "xmin", current_bounds["xmin"]
            )
            y_center = (current_bounds["ymin"] + current_bounds["ymax"]) / 2
            y_half_range = (x_range * self.height) / (self.width * 2)
            bounds["ymin"] = y_center - y_half_range
            bounds["ymax"] = y_center + y_half_range
        elif (
            "ymin" in bounds
            and "ymax" in bounds
            and ("xmin" not in bounds or "xmax" not in bounds)
        ):
            # If x range is not specified, calculate it according to aspect ratio
            current_bounds = self.get_mathBounds()
            y_range = bounds.get("ymax", current_bounds["ymax"]) - bounds.get(
                "ymin", current_bounds["ymin"]
            )
            x_center = (current_bounds["xmin"] + current_bounds["xmax"]) / 2
            x_half_range = (y_range * self.width) / (self.height * 2)
            bounds["xmin"] = x_center - x_half_range
            bounds["xmax"] = x_center + x_half_range

        import json

        # Convert values containing NumPy types to Python standard types
        converted_bounds = {}
        for key, value in bounds.items():
            # Convert NumPy types and other numeric types to float
            if hasattr(value, "item"):  # NumPy scalar
                converted_bounds[key] = float(value.item())
            else:
                converted_bounds[key] = float(value)

        # Convert to JavaScript format with JSON
        bounds_js = json.dumps(converted_bounds)

        self.execute_js(
            f"""
                window.Calc.setMathBounds({bounds_js});
            """,
            update_display=update_display,
        )

    def get_mathBounds(self) -> Dict:
        """Get Desmos display range"""
        bound = self.execute_js(
            """
                window.Calc.getState().graph.viewport;
            """,
            update_display=False,
        )
        # Adjust y-axis range according to aspect ratio
        return (
            self.is3D
            and bound
            or {
                "xmin": bound["xmin"],
                "xmax": bound["xmax"],
                "ymin": (bound["ymin"] + bound["ymax"]) / 2
                + (bound["ymin"] - bound["ymax"])
                / 2
                / (bound["ymax"] - bound["ymin"])
                * (self.height / self.width)
                * (bound["xmax"] - bound["xmin"]),
                "ymax": (bound["ymin"] + bound["ymax"]) / 2
                + (-bound["ymin"] + bound["ymax"])
                / 2
                / (bound["ymax"] - bound["ymin"])
                * (self.height / self.width)
                * (bound["xmax"] - bound["xmin"]),
            }
        )

    def set_rotation_to(
        self, z_tip=None, xy_rot=None, update_display: bool = True, **kwargs
    ):
        """
        Set 3D graph rotation angle as absolute value (immediate reflection).
        z_tip, xy_rot are in radians. Either one can be specified.
        """
        if not self.is3D:
            return

        # Get current rotation angle
        current_orientation = self.get_current_orientation()
        z_tip_val = z_tip if z_tip is not None else current_orientation["zTip"]
        xy_rot_val = xy_rot if xy_rot is not None else current_orientation["xyRot"]

        # Same calculation as orientationFromEuler in orientation.ts
        cos_z = math.cos(z_tip_val)
        sin_z = math.sin(z_tip_val)
        cos_xy = math.cos(xy_rot_val)
        sin_xy = math.sin(xy_rot_val)

        m11 = cos_z * sin_xy
        m12 = cos_z * cos_xy
        m13 = -sin_z
        m21 = -cos_xy
        m22 = sin_xy
        m23 = 0
        m31 = sin_z * sin_xy
        m32 = sin_z * cos_xy
        m33 = cos_z
        self.execute_js(
            self._get_set_rotation_js(m11, m12, m13, m21, m22, m23, m31, m32, m33),
            update_display=update_display,
        )

    def set_rotation_by(
        self, z_tip_delta=None, xy_rot_delta=None, update_display: bool = True
    ):
        """
        Set 3D graph rotation angle as relative value (immediate reflection).
        z_tip_delta, xy_rot_delta are in radians. Either one can be specified.
        """
        if not self.is3D:
            return

        # Get current rotation angle
        current_orientation = self.get_current_orientation()
        z_tip_val = current_orientation["zTip"] + (
            z_tip_delta if z_tip_delta is not None else 0
        )
        xy_rot_val = current_orientation["xyRot"] + (
            xy_rot_delta if xy_rot_delta is not None else 0
        )

        # Same calculation as orientationFromEuler in orientation.ts
        cos_z = math.cos(z_tip_val)
        sin_z = math.sin(z_tip_val)
        cos_xy = math.cos(xy_rot_val)
        sin_xy = math.sin(xy_rot_val)

        m11 = cos_z * sin_xy
        m12 = cos_z * cos_xy
        m13 = -sin_z
        m21 = -cos_xy
        m22 = sin_xy
        m23 = 0
        m31 = sin_z * sin_xy
        m32 = sin_z * cos_xy
        m33 = cos_z
        self.execute_js(
            self._get_set_rotation_js(m11, m12, m13, m21, m22, m23, m31, m32, m33),
            update_display=update_display,
        )

    def set_parameter(
        self, parameter_name: str, parameter_value, update_display: bool = True
    ):
        """Set Desmos parameter value by name or ID"""
        exp_id = parameter_name
        expressions = list(
            filter(
                lambda expr: expr.get("id") == exp_id
                or ((expr.get("latex") or "").split("=")[0] == exp_id),
                self.get_expressions(),
            )
        )
        if len(expressions) == 0:
            raise ValueError(
                f"Expression ID or parameter name '{exp_id}' not found in Desmos graph."
            )
        expression = expressions[0]
        latex = expression.get("latex", "")
        parameter_name = ""
        if expression.get("id") == exp_id:
            parameter_name = latex.split("=")[0] if latex else ""
        else:
            parameter_name = exp_id
            exp_id = expression.get("id")

        self.execute_js(
            f"""
                window.Calc.setExpression({{
                    id: "{exp_id}",
                    latex: "{parameter_name}={parameter_value}"
                }});
            """,
            update_display=update_display,
        )

    def action_single_step(self, exp_id: str, update_display: bool = True):
        """Execute single step action in Desmos"""
        self.execute_js(
            f"""
                window.Calc.controller.dispatch({{
                    type: 'action-single-step',
                    id: "{exp_id}"
                }});
                new Promise((resolve) => {{
                    window.Calc.controller.evaluator.notifyWhenSynced(resolve);
                }});
            """,
            update_display=update_display,
        )

    def action_multi_step(self, exp_id: str, steps: int, update_display: bool = True):
        """Execute multi-step action in Desmos"""
        if steps <= 0:
            return
        else:
            # Set update_display=True only for the last step
            for i in range(steps - 1):
                self.execute_js(
                    f"""
                        window.Calc.controller.dispatch({{
                            type: 'action-single-step',
                            id: "{exp_id}",
                        }});
                        new Promise((resolve) => {{
                            window.Calc.controller.evaluator.notifyWhenSynced(resolve);
                        }});
                    """,
                    update_display=False,
                )
            self.action_single_step(exp_id, update_display=update_display)

    def update_display(self):
        """Update display (create Mobject from file path or data)"""
        if self.use_cache:
            self._update_display_with_cache()
        else:
            self._update_display_direct()

    def _update_display_with_cache(self):
        """Update display with cache"""
        # Get screenshot file path
        image_path = self._create_screenshot()

        # Create Mobject directly from file path
        if self.img_format == "svg":
            new_image = SVGMobject(image_path)
        else:
            new_image = ImageMobject(image_path)

        self._replace_current_image(new_image)

    def _update_display_direct(self):
        """Update display without cache"""
        # Get screenshot data directly
        screenshot_data = self._create_screenshot()

        if self.img_format == "svg":
            # Create temporary file for SVG
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".svg", delete=False
            ) as temp_file:
                temp_file.write(screenshot_data)
                temp_file.flush()
                new_image = SVGMobject(
                    temp_file.name, height=self.graph_height, width=self.graph_width
                )
                os.unlink(temp_file.name)  # Delete temporary file
        else:
            # Create directly from data for PNG
            header, base64_data = screenshot_data.split(",", 1)
            screenshot_bytes = base64.b64decode(base64_data)
            pil_image = Image.open(io.BytesIO(screenshot_bytes))
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            numpy_array = np.array(pil_image)
            new_image = ImageMobject(numpy_array)

        self._replace_current_image(new_image)

    def _replace_current_image(self, new_image):
        """Common process to replace current image with new image"""
        # Remove old image
        if self.current_image is not None:
            # Replace old image with new image
            self.current_image.become(
                new_image,
                match_height=True,
                match_width=True,
                match_depth=True,
                match_center=True,
                stretch=True,
            )
        else:
            self.current_image = new_image
            self.add(self.current_image)

    def add_updater(self, updater: Callable):
        """Add updater"""
        if self.current_image is not None:
            self.current_image.add_updater(updater)

    def remove_updater(self, updater: Callable):
        """Remove updater"""
        if self.current_image is not None:
            self.current_image.remove_updater(updater)

    def get_updaters(self):
        """Get updaters"""
        if self.current_image is not None:
            return self.current_image.get_updaters()
        return []

    def animate_parameter(
        self, name, start_value, end_value, update_display: bool = True, **kwargs
    ):
        """Animate parameter"""
        return DesmosParameterAnimation(
            self, name, start_value, end_value, update_display=update_display, **kwargs
        )

    def animate_translation(
        self, dx=0, dy=0, dz=0, update_display: bool = True, **kwargs
    ):
        """Create animation to translate the graph"""
        current_bounds = self.get_mathBounds()

        # Get current boundary values
        start_bounds = {
            "xmin": current_bounds.get("xmin", -10),
            "xmax": current_bounds.get("xmax", 10),
            "ymin": current_bounds.get("ymin", -10),
            "ymax": current_bounds.get("ymax", 10),
        }

        # Add z-axis for 3D
        if self.is3D:
            start_bounds["zmin"] = current_bounds.get("zmin", -10)
            start_bounds["zmax"] = current_bounds.get("zmax", 10)

        # Calculate boundary values after translation
        end_bounds = {
            "xmin": start_bounds["xmin"] + dx,
            "xmax": start_bounds["xmax"] + dx,
            "ymin": start_bounds["ymin"] + dy,
            "ymax": start_bounds["ymax"] + dy,
        }

        if self.is3D:
            end_bounds["zmin"] = start_bounds["zmin"] + dz
            end_bounds["zmax"] = start_bounds["zmax"] + dz

        return DesmosTranslationAnimation(
            self, start_bounds, end_bounds, update_display=update_display, **kwargs
        )

    def animate_bounds_transition(
        self, target_bounds, update_display: bool = True, **kwargs
    ):
        """Create animation to transition to specified mathBounds"""
        current_bounds = self.get_mathBounds()

        # Get current boundary values
        start_bounds = {
            "xmin": current_bounds.get("xmin", -10),
            "xmax": current_bounds.get("xmax", 10),
            "ymin": current_bounds.get("ymin", -10),
            "ymax": current_bounds.get("ymax", 10),
        }

        if self.is3D:
            start_bounds["zmin"] = current_bounds.get("zmin", -10)
            start_bounds["zmax"] = current_bounds.get("zmax", 10)

        return DesmosTranslationAnimation(
            self, start_bounds, target_bounds, update_display=update_display, **kwargs
        )

    def animate_zoom(
        self,
        center_x=0,
        center_y=0,
        center_z=0,
        scale=1,
        update_display: bool = True,
        **kwargs,
    ):
        """Create animation to zoom the graph centered at specified point"""
        current_bounds = self.get_mathBounds()
        width = current_bounds["xmax"] - current_bounds["xmin"]
        height = current_bounds["ymax"] - current_bounds["ymin"]
        new_width = width / scale
        new_height = height / scale
        target_bounds = {
            "xmin": center_x - new_width / 2,
            "xmax": center_x + new_width / 2,
            "ymin": center_y - new_height / 2,
            "ymax": center_y + new_height / 2,
        }
        if self.is3D:
            depth = current_bounds["zmax"] - current_bounds["zmin"]
            new_depth = depth / scale
            target_bounds["zmin"] = center_z - new_depth / 2
            target_bounds["zmax"] = center_z + new_depth / 2
        return DesmosTranslationAnimation(
            self, current_bounds, target_bounds, update_display=update_display, **kwargs
        )

    def animate_rotation(
        self, z_tip_delta=0, xy_rot_delta=0, update_display: bool = True, **kwargs
    ):
        """Create animation to rotate 3D graph (relative angle specification)"""
        if not self.is3D:
            raise ValueError("Rotation animation is only available for 3D graphs")

        return DesmosRotationAnimation(
            self,
            end_z_tip=z_tip_delta,  # Used as relative value
            end_xy_rot=xy_rot_delta,  # Used as relative value
            relative=True,  # Relative rotation flag
            update_display=update_display,
            **kwargs,
        )

    def animate_rotation_to(
        self, z_tip=None, xy_rot=None, update_display: bool = True, **kwargs
    ):
        """Create animation to rotate 3D graph to specified angle (absolute angle specification)"""
        if not self.is3D:
            raise ValueError("Rotation animation is only available for 3D graphs")

        return DesmosRotationAnimation(
            self,
            end_z_tip=z_tip,
            end_xy_rot=xy_rot,
            update_display=update_display,
            **kwargs,
        )

    def get_current_orientation(self) -> Dict:
        """Get current 3D rotation angle"""
        if not self.is3D:
            return {"zTip": 0, "xyRot": 0}

        # Get current rotation matrix from Desmos worldRotation3D
        matrix_elements = self.execute_js(
            """
            Calc.controller.grapher3d.controls.worldRotation3D.elements;
            """,
            update_display=False,
        )

        if matrix_elements and len(matrix_elements) >= 9:
            # Same calculation as eulerFromOrientation in matrix3.ts
            z_tip = math.atan2(-matrix_elements[6], matrix_elements[8])
            xy_rot = math.atan2(matrix_elements[4], -matrix_elements[1])
            if xy_rot < 0:
                xy_rot += 2 * math.pi
            return {"zTip": z_tip, "xyRot": xy_rot}
        else:
            return {"zTip": 0, "xyRot": 0}

    def _get_set_rotation_js(self, m11, m12, m13, m21, m22, m23, m31, m32, m33) -> str:
        """Generate JavaScript code to set 3D graph rotation"""

        # TODO: Sometimes graph doesn't appear in screenshot without getState(). If there's a lighter and more reliable method than getState(), switch to that.
        return f"""
            const matrix = Calc.controller.grapher3d.controls.worldRotation3D.clone().set(
                {m11}, {m12}, {m13},
                {m21}, {m22}, {m23},
                {m31}, {m32}, {m33}
            );
            Calc.controller.grapher3d.controls.worldRotation3D = matrix;
            Calc.controller.grapher3d.viewportController.animateToOrientation(matrix);
            Calc.controller.grapher3d.transition.duration = 0;
            Calc.getState();
            """

    def cleanup(self):
        # Close browser
        try:
            if hasattr(self, "browser"):
                asyncio.run(self.browser.close())
            if hasattr(self, "playwright"):
                asyncio.run(self.playwright.stop())
        except Exception:
            pass

    def __deepcopy__(self, memo):
        """Exclude Playwright-related objects in deepcopy"""
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Exclude Playwright-related objects
            if k in ["playwright", "browser", "page"]:
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __del__(self):
        self.cleanup()


class DesmosParameterAnimation(Animation):
    """Desmos parameter animation"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        exp_id: str,
        start_value: float,
        end_value: float,
        update_display: bool = True,
        **kwargs,
    ):
        super().__init__(desmos_graph, **kwargs)
        self.desmos_graph = desmos_graph
        self.exp_id = exp_id
        self.start_value = start_value
        self.end_value = end_value
        self.update_display = update_display
        expressions = list(
            filter(
                lambda expr: expr.get("id") == self.exp_id
                or ((expr.get("latex") or "").split("=")[0] == self.exp_id),
                self.desmos_graph.get_expressions(),
            )
        )
        if len(expressions) == 0:
            raise ValueError(
                f"Expression ID or parameter name '{self.exp_id}' not found in Desmos graph."
            )
        expression = expressions[0]
        self.latex = expression.get("latex", "")
        if expression.get("id") == self.exp_id:
            self.parameter_name = self.latex.split("=")[0] if self.latex else ""
        else:
            self.parameter_name = self.exp_id
            self.exp_id = expression.get("id")

    def interpolate_mobject(self, alpha):
        """Update parameter according to animation progress"""
        # Manually apply easing (rate_func) since it doesn't work for some reason
        current_value = self.start_value + self.rate_func(alpha) * (
            self.end_value - self.start_value
        )
        self.desmos_graph.set_expression(
            {"id": self.exp_id, "latex": f"{self.parameter_name}={current_value}"},
            update_display=self.update_display,
        )


class DesmosActionAnimation(Animation):
    """Desmos action animation"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        exp_id: str,
        steps: int = 1,  # Number of executions
        update_display: bool = True,
        **kwargs,
    ):
        super().__init__(desmos_graph, **kwargs)
        self.desmos_graph = desmos_graph
        self.exp_id = exp_id
        self.steps = steps
        self.update_display = update_display
        self.last_step = -1

    def interpolate_mobject(self, alpha):
        """Execute action multiple times according to animation progress"""
        # Calculate which step based on progress
        current_step = int(alpha * self.steps)
        # Execute action when new step is reached
        while self.last_step < current_step:
            self.last_step += 1
            if self.last_step < self.steps:
                self.desmos_graph.action_single_step(
                    self.exp_id, update_display=self.update_display
                )


class DesmosTranslationAnimation(Animation):
    """Desmos graph translation animation (using mathBounds)"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        start_bounds: Dict,  # {xmin, xmax, ymin, ymax, zmin, zmax}
        end_bounds: Dict,  # {xmin, xmax, ymin, ymax, zmin, zmax}
        update_display: bool = True,
        **kwargs,
    ):
        super().__init__(desmos_graph, **kwargs)
        self.desmos_graph = desmos_graph
        self.start_bounds = start_bounds
        self.end_bounds = end_bounds
        self.update_display = update_display

    def interpolate_mobject(self, alpha):
        """Update mathBounds according to animation progress"""
        # Manually apply easing (rate_func)
        eased_alpha = self.rate_func(alpha)

        # Linear interpolation for each axis value
        current_bounds = {}
        for key in self.start_bounds:
            if key in self.end_bounds:
                start_val = self.start_bounds[key]
                end_val = self.end_bounds[key]
                current_bounds[key] = start_val + eased_alpha * (end_val - start_val)

        # Update mathBounds
        self.desmos_graph.set_mathBounds(
            current_bounds, update_display=self.update_display
        )


class DesmosRotationAnimation(Animation):
    """Desmos 3D graph rotation animation (using Euler angles)"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        start_z_tip: float | None = None,  # Starting zTip angle (radians)
        end_z_tip: float | None = None,  # Ending zTip angle (radians)
        start_xy_rot: float | None = None,  # Starting xyRot angle (radians)
        end_xy_rot: float | None = None,  # Ending xyRot angle (radians)
        relative: bool = False,  # Whether relative rotation
        update_display: bool = True,
        **kwargs,
    ):
        super().__init__(desmos_graph, **kwargs)
        self.desmos_graph = desmos_graph
        self.update_display = update_display

        # Get current rotation angle
        current_orientation = desmos_graph.get_current_orientation()

        self.start_z_tip = (
            start_z_tip if start_z_tip is not None else current_orientation["zTip"]
        )
        self.start_xy_rot = (
            start_xy_rot if start_xy_rot is not None else current_orientation["xyRot"]
        )

        if relative:
            # For relative rotation, add to current angle
            self.end_z_tip = self.start_z_tip + (
                end_z_tip if end_z_tip is not None else 0
            )
            self.end_xy_rot = self.start_xy_rot + (
                end_xy_rot if end_xy_rot is not None else 0
            )
        else:
            # For absolute rotation
            self.end_z_tip = (
                end_z_tip if end_z_tip is not None else current_orientation["zTip"]
            )
            self.end_xy_rot = (
                end_xy_rot if end_xy_rot is not None else current_orientation["xyRot"]
            )

    def _atan2_positive(self, y: float, x: float) -> float:
        """Normalize atan2 result to 0~2π range"""
        a = math.atan2(y, x)
        if a < 0:
            a += 2 * math.pi
        return a

    def _set_orientation_from_euler(self, z_tip: float, xy_rot: float):
        """Set 3D rotation from Euler angles"""
        if not self.desmos_graph.is3D:
            return

        # Same calculation as orientationFromEuler in orientation.ts
        # zTip rotation matrix
        cos_z = math.cos(z_tip)
        sin_z = math.sin(z_tip)

        # xyRot rotation matrix
        cos_xy = math.cos(xy_rot)
        sin_xy = math.sin(xy_rot)

        # Rotation matrix multiplication result (as per orientationFromEuler comment)
        m11 = cos_z * sin_xy
        m12 = cos_z * cos_xy
        m13 = -sin_z
        m21 = -cos_xy
        m22 = sin_xy
        m23 = 0
        m31 = sin_z * sin_xy
        m32 = sin_z * cos_xy
        m33 = cos_z

        # Set rotation matrix in Desmos
        self.desmos_graph.execute_js(
            self.desmos_graph._get_set_rotation_js(
                m11, m12, m13, m21, m22, m23, m31, m32, m33
            ),
            update_display=False,
        )

    def interpolate_mobject(self, alpha):
        """Update 3D rotation according to animation progress"""
        if not self.desmos_graph.is3D:
            return

        # Manually apply easing (rate_func)
        eased_alpha = self.rate_func(alpha)

        # Linear interpolation of angles
        current_z_tip = self.start_z_tip + eased_alpha * (
            self.end_z_tip - self.start_z_tip
        )
        current_xy_rot = self.start_xy_rot + eased_alpha * (
            self.end_xy_rot - self.start_xy_rot
        )

        # Set 3D rotation
        self._set_orientation_from_euler(current_z_tip, current_xy_rot)

        # Update display
        self.desmos_graph.update_display() if self.update_display else None
