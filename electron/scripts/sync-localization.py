"""Import the native captions and palettes; keep Electron-specific wording explicit."""
import ast
import json
from pathlib import Path
app=Path(__file__).resolve().parents[1]
repository=app.parent
catalog={}
for relative in ('python/ui/localization.py','python/ui/workspace/localization.py'):
    for node in ast.parse((repository/relative).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_ZH_CN' for target in node.targets):
            catalog.update(ast.literal_eval(node.value))
catalog.update(json.loads((app/'src/locales/electron-zh.json').read_text()))
(app/'src/locales/zh.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
(app/'src/locales/themes.json').write_text((repository/'python/design/resources/themes.json').read_text())
print(f'Synchronized {len(catalog)} captions and native dark/light palettes.')
