# Add some custom arrow tips

from manim import *


class ArrowSharpTriangleTip(ArrowTip, Triangle):
    r"""Sharp triangular arrow tip."""

    def __init__(
        self,
        fill_opacity: float = 0,
        stroke_width: float = 3,
        length: float = DEFAULT_ARROW_TIP_LENGTH,
        width: float = DEFAULT_ARROW_TIP_LENGTH * 0.7,
        start_angle: float = PI,
        **kwargs,
    ) -> None:
        Triangle.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            start_angle=start_angle,
            **kwargs,
        )
        self.width = width

        self.stretch_to_fit_width(length)
        self.stretch_to_fit_height(width)


class ArrowSharpTriangleFilledTip(ArrowSharpTriangleTip):
    r"""Sharp triangular arrow tip with filled tip."""

    def __init__(
        self, fill_opacity: float = 1, stroke_width: float = 0, **kwargs
    ) -> None:
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)
