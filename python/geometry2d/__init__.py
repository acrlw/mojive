"""Pure 2D curves, contours and geometric queries, without rendering dependencies."""

from .compiler import Contour2D as Contour2D
from .compiler import flatten_path as flatten_path
from .mesh import Mesh2D as Mesh2D
from .mesh import compile_fill as compile_fill
from .mesh import compile_stroke as compile_stroke
from .mesh import tessellate_contours as tessellate_contours
from .path import Path2D as Path2D
from .path import PathBuilder2D as PathBuilder2D
from .style import Affine2D as Affine2D
from .style import StrokeStyle as StrokeStyle
