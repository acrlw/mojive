"""Keep view-cube drawing and hit geometry inside compact workspace viewports."""

from mojive.ui.viewcube import ViewCube

from .design import DESIGN


class WorkspaceViewCube(ViewCube):
    """Scale the orientation widget and its hit regions together in compact viewports."""

    def __init__(self, selection_padding):
        super().__init__(selection_padding)
        self.reference_scale = 1.0
        self.preview_active = False
        self.workspace_scale = 1.0

    def widget_center(self, rect, style_scale):
        from mojive.ui.viewcube import BACKDROP_RADIUS_PT

        radius = BACKDROP_RADIUS_PT * style_scale
        margin = DESIGN.navigation_margin * self.workspace_scale
        return rect[0] + rect[2] - margin - radius, rect[1] + margin + radius

    def scale_for(self, rect, style_scale):
        from mojive.ui import viewcube

        if self.preview_active and rect[2] < 180 * style_scale and rect[3] < 300 * style_scale:
            return 0.0
        margin = DESIGN.navigation_margin * style_scale
        # Reserve the left controls before scaling glyphs and hit radii together.
        diameter = max(
            0,
            min(
                DESIGN.navigation_size * style_scale,
                rect[2] - margin - 54 * style_scale,
                rect[3] - margin - 8 * style_scale,
            ),
        )
        return diameter / (2 * viewcube.BACKDROP_RADIUS_PT)

    def update(self, cam, rect, cursor, style_scale, *, enabled=True):
        self.workspace_scale = style_scale
        self.reference_scale = self.scale_for(rect, style_scale)
        if self.reference_scale <= 0:
            self._balls = []
            self._hover = None
            self._origin_hovered = False
            return None
        return super().update(cam, rect, cursor, self.reference_scale, enabled=enabled)

    def draw(self, overlay, style_scale=1.0, **kwargs):
        super().draw(overlay, self.reference_scale, **kwargs)
