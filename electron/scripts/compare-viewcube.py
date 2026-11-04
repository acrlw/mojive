"""Compare matched production captures without conflating glyph and geometry rasterization."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter

out = Path(__file__).resolve().parents[2] / 'output/electron/verification'
metadata = json.loads((out / 'viewcube-native-capture.json').read_text())
dpr = metadata['dpr']
report = {'capture_scale': dpr, 'logical_widget_size': metadata['logical_widget_size'],
          'note': 'Geometry is the widget circle excluding a 2-point dilation of bright neutral glyph pixels. Chromium and FreeType use different text antialiasing.', 'glyph_mask': {'minimum_channel': 150, 'maximum_channel_spread': 32, 'dilation_points': 2}, 'views': {}}
for name in metadata['views']:
    native = np.asarray(Image.open(out / f'viewcube-native-{name}.png').convert('RGB'), dtype=np.float64)
    web = np.asarray(Image.open(out / f'viewcube-web-{name}.png').convert('RGB'), dtype=np.float64)
    assert native.shape == web.shape, (name, native.shape, web.shape)
    difference = np.abs(native - web)
    text_mask = np.zeros(native.shape[:2], dtype=bool)
    for pixels in (native, web):
        text_mask |= (pixels.min(axis=2) > 150) & (np.ptp(pixels, axis=2) < 32)
    diameter = 2 * round(2 * dpr) + 1
    text_mask = np.asarray(Image.fromarray(text_mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(diameter))) > 0
    y, x = np.indices(native.shape[:2])
    region = ((x - native.shape[1] / 2) ** 2 + (y - native.shape[0] / 2) ** 2 < (47 * dpr) ** 2) & ~text_mask
    def errors(values):
        return {'mean_absolute_channel_error': float(values.mean()), 'max_channel_error': float(values.max()),
                'fraction_pixels_above_8': float((values.max(axis=-1) > 8).mean())}
    report['views'][name] = {**errors(difference), 'geometry_region_excluding_dilated_light_text': errors(difference[region])}
(out / 'viewcube-raster-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
for name, result in report['views'].items():
    maximum = result['geometry_region_excluding_dilated_light_text']['max_channel_error']
    assert maximum <= 8, f'{name}: geometry channel difference {maximum}/255 exceeds 8/255'
print('Five matched ViewCube geometry captures remain within 8/255 per channel.')
