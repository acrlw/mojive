"""Capture Mojive's production ViewCube with the Electron specification and MiSans."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from imgui_bundle import imgui
from mojive.app.ui.window import create_window
from mojive.ui.window import WindowConfig
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.viewcube import ViewCube
from mojive.types import CameraView
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/electron/verification'
OUT.mkdir(parents=True, exist_ok=True)
views = [('oblique',(1,1,.6)),('front',(0,-1,0)),('top',(0,-.001745,1)),('back-fade',(-.84,.542586,0)),('neck-limit',(.904517,.426471,0))]
window = create_window(WindowConfig(width=180,height=180,vsync=False,docking=False,ini_path='',show_on_start=False,ui_scale=1), 'opengl')
cube=ViewCube(); scale=90/91
try:
    config=imgui.ImFontConfig()
    config.extra_size_scale=1326/1000 # workspace/fonts.py:font_em_ratio(MiSans)
    font=imgui.get_io().fonts.add_font_from_file_ttf(str(ROOT/'python/text/resources/fonts/MiSans-Regular.ttf'),14,config)
    images=[]
    for name,eye in views:
        for _ in range(3):
            window.begin_frame()
            imgui.push_font(font,14)
            draw=ImguiDraw2D(imgui.get_foreground_draw_list())
            width,height=window.size_points
            draw.rect_filled((0,0),(width,height),(30/255,33/255,37/255,1))
            reach=(34+9.5+10)*scale
            rect=(0,height*.5-reach,width*.5+reach,height)
            cube.update(CameraView(eye=np.array(eye),target=np.zeros(3),up=np.array([0,0,1])),rect,(-1000,-1000),scale)
            cube.draw(draw,scale)
            imgui.pop_font()
            pixels=window.end_frame(readback=True)[::-1].copy()
        image=Image.fromarray(pixels)
        image.save(OUT/f'viewcube-native-{name}.png')
        images.append(pixels)
    Image.fromarray(np.concatenate(images,axis=1)).save(OUT/'viewcube-native-gallery.png')
    (OUT/'viewcube-native-capture.json').write_text(json.dumps({'dpr':window.pixel_scale,'logical_cell_size':180,'logical_widget_size':90,'views':[name for name,_ in views]},indent=2))
    print('Native ViewCube captured at scale',window.pixel_scale)
finally:
    window.close()
