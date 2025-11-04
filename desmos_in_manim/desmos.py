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
    """DesmosグラフをManimで表示"""

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
            Desmosの状態(JSON文字列)。
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
            取得する画像フォーマット。'png'または'svg'。3Dは'svg'非対応。
        use_cache: bool = False
            画像をキャッシュするかどうか。Falseの場合、get_state()などの処理を省略できるため逆に高速になる可能性もある。
            Trueでもstateの計算自体は行い、あくまでもスクリーンショットを撮る工程を省略する。
        """
        super().__init__(**kwargs)

        # キャッシュ設定
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir = Path("media/desmos")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # 現在の画像
        self.current_image = None
        self.graph_width = width
        self.graph_height = height
        self.background_color = background_color
        self.is3D = is3D
        self.calc_options = calc_options
        # 3DならばbackgroundColorはcalc_optionsに設定すれば反映される
        # 2Dの場合は不等式で塗りつぶす必要がある
        if background_color is not None:
            if self.calc_options is None:
                self.calc_options = {}
            self.calc_options["backgroundColor"] = background_color
        self.graph_settings = graph_settings
        self.img_format = img_format if not is3D else "png"
        # TODO: svgは上手く表示できないので一旦無効化
        self.img_format = "png"
        # isPlayingはfalseにする
        asyncio.run(
            self._init_graph(
                re.sub(r".isPlaying.: *true", '"isPlaying": false', state or "")
            )
        )
        self.update_display()
        atexit.register(self.cleanup)

    async def _init_graph(self, state: str):
        self.playwright = await async_playwright().start()
        # 3DはGPUを有効化しないと遅い
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--enable-gpu"],
        )
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 600, "height": 400})
        # これ以上長いhtmlはdata URLで渡せない可能性あり
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
        """現在のDesmosの状態からキャッシュキーを生成"""
        try:
            # 現在の状態を取得
            state = self.get_state()

            # キャッシュキーに影響する要素を組み合わせ
            cache_data = {
                "state": state,
                "width": self.graph_width,
                "height": self.graph_height,
                "is3D": self.is3D,
                "img_format": self.img_format,
                "calc_options": self.calc_options,
                "graph_settings": self.graph_settings,
            }

            # JSON文字列にしてハッシュ化
            cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(cache_str.encode("utf-8")).hexdigest()[
                :16
            ]  # 16文字に短縮
        except Exception:
            # エラーの場合はタイムスタンプベースのキーを返す
            import time

            return f"fallback_{int(time.time())}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """キャッシュファイルのパスを取得"""
        # self.cache_dir が None の場合、fallback として一時ディレクトリ内にキャッシュディレクトリを作成する
        if self.cache_dir is None:
            # tempfile はファイルの先頭でインポート済み
            fallback_dir = Path(tempfile.gettempdir()) / "desmos"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = fallback_dir
        else:
            cache_dir = self.cache_dir

        extension = "svg" if self.img_format == "svg" else "png"
        return cache_dir / f"{cache_key}.{extension}"

    def _load_from_cache(self, cache_key: str):
        """キャッシュファイルが存在する場合、そのパスを返す"""
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            return str(cache_path)  # ファイルパスを文字列として返す

        return None

    def _save_to_cache(self, cache_key: str, data, is_data_url: bool = False):
        """データをキャッシュファイルに保存"""
        cache_path = self._get_cache_path(cache_key)

        try:
            if self.img_format == "svg":
                # SVG文字列をファイルに保存
                cache_path.write_text(data, encoding="utf-8")
            else:
                # data URLの場合はbase64デコードして保存
                if is_data_url:
                    header, base64_data = data.split(",", 1)
                    screenshot_bytes = base64.b64decode(base64_data)
                    cache_path.write_bytes(screenshot_bytes)
                else:
                    # NumPy配列をPNG画像として保存
                    pil_image = Image.fromarray(data)
                    pil_image.save(cache_path, "PNG")
        except Exception as e:
            print(f"キャッシュ保存に失敗: {e}")

    def clear_cache(self):
        """キャッシュディレクトリをクリア"""
        try:
            if self.cache_dir is None:
                print("キャッシュディレクトリが設定されていません。")
                return
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
            print(f"キャッシュをクリアしました: {self.cache_dir}")
        except Exception as e:
            print(f"キャッシュクリアに失敗: {e}")

    def get_cache_info(self):
        """キャッシュの情報を表示"""
        if self.cache_dir is None:
            print("キャッシュディレクトリが設定されていません。")
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
        """JavaScriptコードを実行"""
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.page.evaluate(script))
        if update_display:
            self.update_display()
        return res

    def update_calc_options(self, options: Dict):
        """Desmosのcalc_optionsを更新"""
        for key, value in options.items():
            self.execute_js(
                f"""
                    Calc.controller.graphSettings.config.{key} = {str(value).lower() if isinstance(value, bool) else value};
                """
            )

    def update_graph_settings(self, settings: Dict):
        """Desmosのgraph_settingsを更新"""
        for key, value in settings.items():
            self.execute_js(
                f"""
                    Calc.controller.graphSettings.{key} = {str(value).lower() if isinstance(value, bool) else value};
                """
            )

    def _create_screenshot(self):
        """スクリーンショット取得してファイルパスまたはデータとして返す"""
        if self.use_cache:
            return self._create_screenshot_with_cache()
        else:
            return self._create_screenshot_direct()

    def _create_screenshot_with_cache(self):
        """キャッシュ機能付きスクリーンショット取得"""
        # stateからキャッシュキーを生成
        cache_key = self._generate_cache_key()

        # キャッシュから読み込みを試行
        cached_path = self._load_from_cache(cache_key)
        if cached_path is not None:
            return cached_path

        # キャッシュが無い場合は新規にスクリーンショットを取得
        data = self._get_screenshot_data()

        # キャッシュに保存
        if self.img_format == "svg":
            self._save_to_cache(cache_key, data)
        else:
            self._save_to_cache(cache_key, data, is_data_url=True)

        # 保存されたファイルのパスを返す
        return str(self._get_cache_path(cache_key))

    def _create_screenshot_direct(self):
        """キャッシュなし直接スクリーンショット取得"""
        # 直接スクリーンショットを取得してデータを返す
        return self._get_screenshot_data()

    def _get_screenshot_data(self):
        """スクリーンショットデータを取得（共通処理）"""
        if self.is3D:
            # 3Dは正方形で
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
        """Desmosを空白に設定"""
        self.execute_js(
            """
                window.Calc.setBlank();
            """
        )

    def set_state(self, state: str, update_display: bool = True):
        """Desmosの状態を設定"""
        self.execute_js(
            f"""
                window.Calc.setState({state});
            """,
            update_display=update_display,
        )

    def get_state(self) -> Dict:
        """Desmosの状態を取得"""
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
        """Desmosの式を設定"""
        self.execute_js(
            f"""
                window.Calc.setExpression(
                    {expression}
                );
            """,
            update_display=update_display,
        )

    def set_expressions(self, expressions: List[Dict], update_display: bool = True):
        """Desmosの式リストを設定"""
        self.execute_js(
            f"""
                window.Calc.setExpressions(
                    {expressions}
                );
            """,
            update_display=update_display,
        )

    def get_expressions(self) -> List[Dict]:
        """Desmosの式リストを取得"""
        return self.execute_js(
            """
                window.Calc.getExpressions();
            """,
            update_display=False,
        )

    def set_mathBounds(self, bounds: Dict, update_display: bool = True):
        """Desmosの表示範囲を設定"""
        import json

        # NumPy型を含む値をPythonの標準型に変換
        converted_bounds = {}
        for key, value in bounds.items():
            # NumPy型やその他の数値型をfloatに変換
            if hasattr(value, "item"):  # NumPy scalar
                converted_bounds[key] = float(value.item())
            else:
                converted_bounds[key] = float(value)

        # JSONで確実にJavaScript形式に変換
        bounds_js = json.dumps(converted_bounds)

        self.execute_js(
            f"""
                window.Calc.setMathBounds({bounds_js});
            """,
            update_display=update_display,
        )

    def get_mathBounds(self) -> Dict:
        """Desmosの表示範囲を取得"""
        bound = self.execute_js(
            """
                window.Calc.getState().graph.viewport;
            """,
            update_display=False,
        )
        # アスペクトに合わせてy軸の範囲を調整
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
        3Dグラフの回転角度を絶対値で設定する（即時反映）。
        z_tip, xy_rotはラジアン。どちらか一方だけ指定も可。
        """
        if not self.is3D:
            return

        # 現在の回転角度を取得
        current_orientation = self.get_current_orientation()
        z_tip_val = z_tip if z_tip is not None else current_orientation["zTip"]
        xy_rot_val = xy_rot if xy_rot is not None else current_orientation["xyRot"]

        # orientation.tsのorientationFromEulerと同じ計算
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
        3Dグラフの回転角度を相対値で設定する（即時反映）。
        z_tip_delta, xy_rot_deltaはラジアン。どちらか一方だけ指定も可。
        """
        if not self.is3D:
            return

        # 現在の回転角度を取得
        current_orientation = self.get_current_orientation()
        z_tip_val = current_orientation["zTip"] + (
            z_tip_delta if z_tip_delta is not None else 0
        )
        xy_rot_val = current_orientation["xyRot"] + (
            xy_rot_delta if xy_rot_delta is not None else 0
        )

        # orientation.tsのorientationFromEulerと同じ計算
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

    def action_single_step(self, exp_id: str, update_display: bool = True):
        """Desmosで単一ステップアクションを実行"""
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
        """Desmosでマルチステップアクションを実行"""
        if steps <= 0:
            return
        else:
            # 最後のみupdate_display=Trueにする
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
        """表示を更新（ファイルパスまたはデータからMobjectを作成）"""
        if self.use_cache:
            self._update_display_with_cache()
        else:
            self._update_display_direct()

    def _update_display_with_cache(self):
        """キャッシュありの表示更新"""
        # スクリーンショットファイルのパスを取得
        image_path = self._create_screenshot()

        # ファイルパスから直接Mobjectを作成
        if self.img_format == "svg":
            new_image = SVGMobject(image_path)
        else:
            new_image = ImageMobject(image_path)

        self._replace_current_image(new_image)

    def _update_display_direct(self):
        """キャッシュなしの表示更新"""
        # 直接スクリーンショットデータを取得
        screenshot_data = self._create_screenshot()

        if self.img_format == "svg":
            # SVGの場合は一時ファイルを作成
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".svg", delete=False
            ) as temp_file:
                temp_file.write(screenshot_data)
                temp_file.flush()
                new_image = SVGMobject(
                    temp_file.name, height=self.graph_height, width=self.graph_width
                )
                os.unlink(temp_file.name)  # 一時ファイルを削除
        else:
            # PNGの場合は直接データから作成
            header, base64_data = screenshot_data.split(",", 1)
            screenshot_bytes = base64.b64decode(base64_data)
            pil_image = Image.open(io.BytesIO(screenshot_bytes))
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            numpy_array = np.array(pil_image)
            new_image = ImageMobject(numpy_array)

        self._replace_current_image(new_image)

    def _replace_current_image(self, new_image):
        """現在の画像を新しい画像に置き換える共通処理"""
        # 古い画像を削除
        if self.current_image is not None:
            # 古い画像を新しい画像に置き換え
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
        """アップデーターを追加"""
        if self.current_image is not None:
            self.current_image.add_updater(updater)

    def remove_updater(self, updater: Callable):
        """アップデーターを削除"""
        if self.current_image is not None:
            self.current_image.remove_updater(updater)

    def get_updaters(self):
        """アップデーターを取得"""
        if self.current_image is not None:
            return self.current_image.get_updaters()
        return []

    def animate_parameter(
        self, name, start_value, end_value, update_display: bool = True, **kwargs
    ):
        """パラメータをアニメーション化"""
        return DesmosParameterAnimation(
            self, name, start_value, end_value, update_display=update_display, **kwargs
        )

    def animate_translation(
        self, dx=0, dy=0, dz=0, update_display: bool = True, **kwargs
    ):
        """グラフを並進移動させるアニメーションを作成"""
        current_bounds = self.get_mathBounds()

        # 現在の境界値を取得
        start_bounds = {
            "xmin": current_bounds.get("xmin", -10),
            "xmax": current_bounds.get("xmax", 10),
            "ymin": current_bounds.get("ymin", -10),
            "ymax": current_bounds.get("ymax", 10),
        }

        # 3Dの場合はz軸も追加
        if self.is3D:
            start_bounds["zmin"] = current_bounds.get("zmin", -10)
            start_bounds["zmax"] = current_bounds.get("zmax", 10)

        # 移動後の境界値を計算
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
        """指定されたmathBoundsに遷移するアニメーションを作成"""
        current_bounds = self.get_mathBounds()

        # 現在の境界値を取得
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
        """グラフを指定した点を中心に拡大縮小させるアニメーションを作成"""
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
        """3Dグラフを回転させるアニメーションを作成（相対角度指定）"""
        if not self.is3D:
            raise ValueError("回転アニメーションは3Dグラフでのみ使用できます")

        return DesmosRotationAnimation(
            self,
            end_z_tip=z_tip_delta,  # 相対値として使用
            end_xy_rot=xy_rot_delta,  # 相対値として使用
            relative=True,  # 相対回転フラグ
            update_display=update_display,
            **kwargs,
        )

    def animate_rotation_to(
        self, z_tip=None, xy_rot=None, update_display: bool = True, **kwargs
    ):
        """3Dグラフを指定角度まで回転させるアニメーションを作成（絶対角度指定）"""
        if not self.is3D:
            raise ValueError("回転アニメーションは3Dグラフでのみ使用できます")

        return DesmosRotationAnimation(
            self,
            end_z_tip=z_tip,
            end_xy_rot=xy_rot,
            update_display=update_display,
            **kwargs,
        )

    def get_current_orientation(self) -> Dict:
        """現在の3D回転角度を取得"""
        if not self.is3D:
            return {"zTip": 0, "xyRot": 0}

        # DesmosのworldRotation3Dから現在の回転行列を取得
        matrix_elements = self.execute_js(
            """
            Calc.controller.grapher3d.controls.worldRotation3D.elements;
            """,
            update_display=False,
        )

        if matrix_elements and len(matrix_elements) >= 9:
            # matrix3.tsのeulerFromOrientationと同じ計算
            z_tip = math.atan2(-matrix_elements[6], matrix_elements[8])
            xy_rot = math.atan2(matrix_elements[4], -matrix_elements[1])
            if xy_rot < 0:
                xy_rot += 2 * math.pi
            return {"zTip": z_tip, "xyRot": xy_rot}
        else:
            return {"zTip": 0, "xyRot": 0}

    def _get_set_rotation_js(self, m11, m12, m13, m21, m22, m23, m31, m32, m33) -> str:
        """3Dグラフの回転を設定するためのJavaScriptコードを生成"""

        # TODO: getState()をしないとたまにスクショにグラフが映らない事がある。getState()より処理が軽く確実な方法があればそちらに変更したい。
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
        # ブラウザを閉じる
        try:
            if hasattr(self, "browser"):
                asyncio.run(self.browser.close())
            if hasattr(self, "playwright"):
                asyncio.run(self.playwright.stop())
        except Exception:
            pass

    def __deepcopy__(self, memo):
        """deepcopyでPlaywright関連のオブジェクトを除外"""
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Playwright関連のオブジェクトは除外
            if k in ["playwright", "browser", "page"]:
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __del__(self):
        self.cleanup()


class DesmosParameterAnimation(Animation):
    """Desmosパラメータアニメーション"""

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
                f"Desmosグラフに式IDまたはパラメータ名 '{self.exp_id}' が見つかりません。"
            )
        expression = expressions[0]
        self.latex = expression.get("latex", "")
        if expression.get("id") == self.exp_id:
            self.parameter_name = self.latex.split("=")[0] if self.latex else ""
        else:
            self.parameter_name = self.exp_id
            self.exp_id = expression.get("id")

    def interpolate_mobject(self, alpha):
        """アニメーション進行に応じてパラメータを更新"""
        # イージング(rate_func)が何故か効かないので手動で適用
        current_value = self.start_value + self.rate_func(alpha) * (
            self.end_value - self.start_value
        )
        self.desmos_graph.set_expression(
            {"id": self.exp_id, "latex": f"{self.parameter_name}={current_value}"},
            update_display=self.update_display,
        )


class DesmosActionAnimation(Animation):
    """Desmosアクションアニメーション"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        exp_id: str,
        steps: int = 1,  # 実行回数
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
        """アニメーション進行に応じてアクションを複数回実行"""
        # 進行度に応じて何回目のステップか計算
        current_step = int(alpha * self.steps)
        # 新しいステップに到達したらアクション実行
        while self.last_step < current_step:
            self.last_step += 1
            if self.last_step < self.steps:
                self.desmos_graph.action_single_step(
                    self.exp_id, update_display=self.update_display
                )


class DesmosTranslationAnimation(Animation):
    """Desmosグラフの移動アニメーション（mathBoundsを使用）"""

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
        """アニメーション進行に応じてmathBoundsを更新"""
        # イージング(rate_func)を手動で適用
        eased_alpha = self.rate_func(alpha)

        # 各座標軸の値を線形補間
        current_bounds = {}
        for key in self.start_bounds:
            if key in self.end_bounds:
                start_val = self.start_bounds[key]
                end_val = self.end_bounds[key]
                current_bounds[key] = start_val + eased_alpha * (end_val - start_val)

        # mathBoundsを更新
        self.desmos_graph.set_mathBounds(
            current_bounds, update_display=self.update_display
        )


class DesmosRotationAnimation(Animation):
    """Desmos 3Dグラフの回転アニメーション（オイラー角を使用）"""

    def __init__(
        self,
        desmos_graph: DesmosGraph,
        start_z_tip: float | None = None,  # 開始時のzTip角度（ラジアン）
        end_z_tip: float | None = None,  # 終了時のzTip角度（ラジアン）
        start_xy_rot: float | None = None,  # 開始時のxyRot角度（ラジアン）
        end_xy_rot: float | None = None,  # 終了時のxyRot角度（ラジアン）
        relative: bool = False,  # 相対回転かどうか
        update_display: bool = True,
        **kwargs,
    ):
        super().__init__(desmos_graph, **kwargs)
        self.desmos_graph = desmos_graph
        self.update_display = update_display

        # 現在の回転角度を取得
        current_orientation = desmos_graph.get_current_orientation()

        self.start_z_tip = (
            start_z_tip if start_z_tip is not None else current_orientation["zTip"]
        )
        self.start_xy_rot = (
            start_xy_rot if start_xy_rot is not None else current_orientation["xyRot"]
        )

        if relative:
            # 相対回転の場合、現在角度に加算
            self.end_z_tip = self.start_z_tip + (
                end_z_tip if end_z_tip is not None else 0
            )
            self.end_xy_rot = self.start_xy_rot + (
                end_xy_rot if end_xy_rot is not None else 0
            )
        else:
            # 絶対回転の場合
            self.end_z_tip = (
                end_z_tip if end_z_tip is not None else current_orientation["zTip"]
            )
            self.end_xy_rot = (
                end_xy_rot if end_xy_rot is not None else current_orientation["xyRot"]
            )

    def _atan2_positive(self, y: float, x: float) -> float:
        """atan2の結果を0～2πの範囲に正規化"""
        a = math.atan2(y, x)
        if a < 0:
            a += 2 * math.pi
        return a

    def _set_orientation_from_euler(self, z_tip: float, xy_rot: float):
        """オイラー角から3D回転を設定"""
        if not self.desmos_graph.is3D:
            return

        # orientation.tsのorientationFromEulerと同じ計算
        # zTip回転行列
        cos_z = math.cos(z_tip)
        sin_z = math.sin(z_tip)

        # xyRot回転行列
        cos_xy = math.cos(xy_rot)
        sin_xy = math.sin(xy_rot)

        # 回転行列の乗算結果（orientationFromEulerのコメント通り）
        m11 = cos_z * sin_xy
        m12 = cos_z * cos_xy
        m13 = -sin_z
        m21 = -cos_xy
        m22 = sin_xy
        m23 = 0
        m31 = sin_z * sin_xy
        m32 = sin_z * cos_xy
        m33 = cos_z

        # Desmosに回転行列を設定
        self.desmos_graph.execute_js(
            self.desmos_graph._get_set_rotation_js(
                m11, m12, m13, m21, m22, m23, m31, m32, m33
            ),
            update_display=False,
        )

    def interpolate_mobject(self, alpha):
        """アニメーション進行に応じて3D回転を更新"""
        if not self.desmos_graph.is3D:
            return

        # イージング(rate_func)を手動で適用
        eased_alpha = self.rate_func(alpha)

        # 角度を線形補間
        current_z_tip = self.start_z_tip + eased_alpha * (
            self.end_z_tip - self.start_z_tip
        )
        current_xy_rot = self.start_xy_rot + eased_alpha * (
            self.end_xy_rot - self.start_xy_rot
        )

        # 3D回転を設定
        self._set_orientation_from_euler(current_z_tip, current_xy_rot)

        # 表示を更新
        self.desmos_graph.update_display() if self.update_display else None
