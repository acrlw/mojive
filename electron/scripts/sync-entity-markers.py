"""Export production scene-marker triangles without maintaining another SVG interpretation."""
import json
from pathlib import Path
from mojive.interaction.scene_icons import scene_icon_geometry
from mojive.ui.scene_entities import HELPER_COLOR, SELECTED_COLOR
root = Path(__file__).resolve().parents[1]
result = {name: scene_icon_geometry(name).tolist() for name in ('camera', 'light')}
result['helper_color'] = HELPER_COLOR.tolist()
result['selected_color'] = SELECTED_COLOR.tolist()
(root / 'src/viewport/entity-masters.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
