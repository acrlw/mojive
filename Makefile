PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
.DEFAULT_GOAL := help

.PHONY: help setup check lint fmt test gpu golden golden-accept parity calibrate gallery gizmo-gallery bench showcase probe reverse viewer canvas lighting text-overlay capture record serve attach pvd snapshot-record snapshot-replay toy-physics adapter-conformance gizmo perturb reflect outline robot mujoco-audit mujoco-visuals mujoco-debug musculoskeletal musculoskeletal-video musculoskeletal-check cameras geom-groups deformables assets backends doctor clean

help:
	@printf '%s\n' \
		'常用：' \
		'  make viewer             默认 MuJoCo 场景' \
		'  make robot             Unitree Go2（首次按需下载）' \
		'  make outline           双击选中与抗锯齿描边' \
		'  make gizmo             2D/3D position/rotation gizmo' \
		'  make gizmo-gallery     输出 2D/3D gizmo 局部放大验收图' \
		'  make perturb           MuJoCo 平移/扭转扰动' \
		'  make text-overlay      GPU 世界空间文字' \
		'  make mujoco-visuals    hfield/site/tendon/contact' \
		'  make mujoco-debug      joint/COM/inertia debug visuals' \
		'  make musculoskeletal   人体肌骨模型 + tendon/keyframe' \
		'  make musculoskeletal-video  300 keyframes → 60 fps MP4' \
		'  make deformables       flex/skin 动态网格' \
		'' \
		'M5/M6 渲染与输出：' \
		'  make lighting          无物理场景的可编辑 spot/point/area 灯光与雾霾' \
		'  make reflect           平面反射' \
		'  make cameras           free/named/正交相机' \
		'  make capture           输出 PNG' \
		'  make record            流式输出 MP4' \
		'  make showcase          生成渲染能力总览图' \
		'' \
		'解耦与远程：' \
		'  make canvas            无物理后端的 3D 画布' \
		'  make toy-physics       无 MuJoCo 的最小物理后端' \
		'  make pvd               一个物理进程 + 两个独立 viewer' \
		'  make snapshot-record   录制远程场景快照' \
		'  make snapshot-replay   脱离物理进程回放快照' \
		'' \
		'自动验收：' \
		'  make check             lint + CPU 测试' \
		'  make gpu               完整真实 OpenGL 测试' \
		'  make doctor            窗口路径 smoke test' \
		'  make mujoco-audit      MuJoCo 可视化覆盖审计' \
		'  make adapter-conformance  adapter 公共契约检查' \
		'' \
		'参数示例：make viewer SCENE=humanoid ARGS="--paused"'

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev,mujoco]"

## lint + 全部单元/集成测试。提交前唯一需要记住的命令（12 §12.4）
check: lint test

lint:
	$(RUFF) check src tests tools
	$(RUFF) format --check src tests tools

fmt:
	$(RUFF) check --fix src tests tools
	$(RUFF) format src tests tools

## 默认不收集需要 GPU / 真实物理世界的用例（12 §12.6）
test:
	$(PYTEST) -q

## 一个文件一个进程：GL/物理库的注册表是进程全局的，连着建几十个世界会崩，
## 而且崩在哪个用例上取决于收集顺序——最像"偶发"的那种失败（12 §12.6）
gpu:
	@for f in $$(ls tests/gpu/test_*.py); do echo "--- $$f"; $(PYTEST) -q -m "gpu or physics" $$f || exit 1; done

## 基准图回归。**只比对，不写基准**——写基准要显式 `make golden-accept`，
## 因为"重新生成之后必须肉眼看一遍"（12 §12.4）：自动生成 + 自动接受等于没有基准
golden:
	$(PY) -m forge_viewer.tools.golden

golden-accept:
	$(PY) -m forge_viewer.tools.golden --accept

## 与参照渲染器同机位对拍，出三联图。参照渲染器跑在**子进程**里
## （mjr_ 要 legacy 上下文、forge 要 core，同进程不能共存——但同一台机器没问题）
parity:
	$(PY) -m forge_viewer.tools.parity

## 拿参照渲染器逐项标定光照（漫反射/头灯/环境光/贴图面/环境光来源）。
## 改 color.py 里任何一个系数之前，先跑它看参照给的是什么
calibrate:
	$(PY) -m forge_viewer.tools.calibrate

## 门禁全是数字与红绿，答不了"现在到底画成什么样"。这条只出图，不判对错（12 §12.4）
gallery:
	$(PY) -m forge_viewer.tools.gallery

gizmo-gallery:
	$(PY) -m forge_viewer.tools.gizmo_gallery $(ARGS)

bench:
	$(PY) -m forge_viewer.tools.bench

showcase:
	$(PY) -m forge_viewer.tools.showcase

## 重跑 GL 能力探测——docs/PLATFORM.md 的每一条都出自它
probe:
	$(PY) tools/probe_gl.py

## 把修复逐条去掉，判据必须立刻红（12 §12.1 第一条纪律）。
## 不红的判据看着是绿的，实际什么都没守住——所以这条要能随时重跑。
reverse:
	$(PY) tools/reverse_verify.py

## 真的把查看器开起来。`make viewer` 用默认场景，换别的写 `make viewer SCENE=arm26`；
## 别的开关走 ARGS，例如 `make viewer SCENE=humanoid ARGS="--paused"`。
## `make assets` 列出本机能加载的全部场景。
SCENE ?= test_scene
ARGS  ?=
viewer:
	$(PY) -m forge_viewer.cli view $(SCENE) $(ARGS)

## 不加载 MuJoCo：程序化场景 + forge 渲染 + 完整 UI。
canvas:
	$(PY) -m forge_viewer.cli canvas $(ARGS)

## 正式的非 MuJoCo adapter：重力、碰撞、播放/暂停/step/Reset 与暂停后位姿编辑。
toy-physics:
	$(PY) -m forge_viewer.cli toy $(ARGS)

ADAPTER ?= toy
CONFORMANCE_ASSET ?=
## 第三方 adapter 的无窗口契约报告。MuJoCo 示例：
## make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=test_scene
adapter-conformance:
	$(PY) -m forge_viewer.cli conformance $(ADAPTER) $(if $(CONFORMANCE_ASSET),--asset $(CONFORMANCE_ASSET),) $(ARGS)

## spot / point / area 三种投影光与雾霾；Hierarchy 选灯后在 Inspector 编辑。
lighting:
	$(PY) -m forge_viewer.cli canvas --demo lighting $(ARGS)

## 世界锚点 + 屏幕偏移/对齐 + depth/always；字体与 ImGui 共用同一份文件和字号。
text-overlay:
	$(PY) -m forge_viewer.cli canvas --demo text $(ARGS)

OUTPUT ?= output/recording.mp4
SCREENSHOT ?= output/capture.png
## 任意分辨率截图。例：make capture SCENE=humanoid ARGS="--width 1920 --height 1080"。
capture:
	$(PY) -m forge_viewer.cli capture $(SCENE) -o $(SCREENSHOT) $(ARGS)

## 流式录制，不把整段视频攒进内存。例：make record SCENE=humanoid ARGS="--frames 120"。
record:
	$(PY) -m forge_viewer.cli record $(SCENE) -o $(OUTPUT) $(ARGS)

PVD_HOST ?= 127.0.0.1
PVD_PORT ?= 47650
PVD_SCENE ?= gizmo
## 无窗口运行物理并发布最新快照；可单独配合一个或多个 `make attach`。
serve:
	$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS)

## 打开一个独立远程窗口。例：make attach ARGS="--debug-view normal --title debug"。
attach:
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS)

## PVD 验收：一个独立物理进程、一个效果窗口、一个 normal debug 窗口。
## 关闭任意 viewer 不影响另一个；两边都关掉后 Make 会收掉 publisher。
pvd:
	@$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS) & server=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge effect" & effect=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge debug" --debug-view normal & debug=$$!; \
	trap 'kill $$effect $$debug $$server 2>/dev/null || true' EXIT INT TERM; \
	wait $$effect; wait $$debug

SNAPSHOT ?= out/session.fvs
## 录制结构变化、物理帧与 debug commands；Ctrl-C 结束并关闭文件。
snapshot-record:
	$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) --record-snapshot $(SNAPSHOT) $(ARGS)

## 通过同一远程协议循环回放，不需要原物理进程仍然存活。
snapshot-replay:
	@$(PY) -m forge_viewer.cli replay $(SNAPSHOT) --host $(PVD_HOST) --port $(PVD_PORT) --loop & server=$$!; \
	trap 'kill $$server 2>/dev/null || true' EXIT INT TERM; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge replay" $(ARGS)

## 原生 gizmo 验收：默认 2D；Settings(F9) 可切 3D。G 平移、R 旋转、T 切 body/world frame。
## 2D/3D 都有平移起终点/连线，以及旋转起始射线/扫角/数值；Ctrl+拖拽仍是物理扰动。
gizmo:
	$(PY) -m forge_viewer.cli view gizmo --paused $(ARGS)

## 物理扰动视觉验收：先点选自由体，Ctrl+左键拖动平移，Ctrl+右键拖动扭转；
## 扭转只显示跟随 body 的二维实线剪影、三轴与抓取点，不画三维方块。
perturb:
	$(PY) -m forge_viewer.cli view gizmo $(ARGS)

## 选中描边（07 §7）。这份场景按描边的三条性质摆：跨几何的整节连杆一条外轮廓、
## 被墙挡住的那半条仍然实心、点旁边那只时别的不许描。
outline:
	$(PY) -m forge_viewer.cli view outline --paused

## 平面反射（14 §M5.2）。这份场景是按判据设计的：一高一低两个球（倒影位置跟着高度走）、
## 一个埋在地板下的球（不许进倒影）、一个开口盒子（绕向翻错了就只剩内壁）。
reflect:
	$(PY) -m forge_viewer.cli view reflection --paused

## Google DeepMind 官方 MuJoCo Menagerie。只稀疏下载选中的模型与公共场景资产，不把模型
## vendoring 进仓库。例：make robot ROBOT=unitree_g1；默认是 Unitree Go2。
ROBOT ?= unitree_go2
MENAGERIE_DIR ?= out/mujoco_menagerie
robot:
	@if [ ! -d "$(MENAGERIE_DIR)/.git" ]; then \
		git clone --depth 1 --filter=blob:none --sparse \
			https://github.com/google-deepmind/mujoco_menagerie.git "$(MENAGERIE_DIR)"; \
	fi
	@git -C "$(MENAGERIE_DIR)" sparse-checkout add assets "$(ROBOT)"
	$(PY) -m forge_viewer.cli view "$(MENAGERIE_DIR)/$(ROBOT)/scene.xml" $(ARGS)

AUDIT_SCENE ?= mujoco_visuals
## 无窗口逐项报告 MuJoCo 可视化覆盖；遇到会被真正跳过的特性就失败。
mujoco-audit:
	$(PY) -m forge_viewer.cli audit $(AUDIT_SCENE) --strict

## heightfield + site + tendon + contact point/force 的可交互验收场景。
mujoco-visuals:
	$(PY) -m forge_viewer.cli view mujoco_visuals \
		--enable-render tendon --enable-render contactpoint --enable-render contactforce $(ARGS)

## MuJoCo joint markers, root subtree COM and body inertia boxes.
mujoco-debug:
	$(PY) -m forge_viewer.cli view joint_types --paused \
		--enable-render joint --enable-render com --enable-render inertia $(ARGS)

MYO_SCENE ?= ../lowerlimb-refactor/lowerlimb-main/assets/models/myo_sim_latest/myo_sim/body/fullbody_kit_9_10_walk_forward_60.xml
MYO_VIDEO ?= output/musculoskeletal-keyframes-60fps.mp4
## 人体肌骨模型：默认暂停，Control 面板可拖动/加载 300 个 keyframe，肌腱路径默认打开。
musculoskeletal:
	$(PY) -m forge_viewer.cli view "$(MYO_SCENE)" --paused $(ARGS)

## 全部 300 个 keyframe，一帧一个姿态；60 fps 正好 5 秒。默认只画模型本来的红色 tendon。
musculoskeletal-video:
	$(PY) -m forge_viewer.cli keyframes "$(MYO_SCENE)" -o "$(MYO_VIDEO)" --fps 60 \
		--camera cam_track $(ARGS)

## 无窗口验证结构、完整帧（含 tendon/sensor/actuator/deformable）与 MuJoCo 可视化审计。
musculoskeletal-check:
	$(PY) -m forge_viewer.cli conformance mujoco --asset "$(MYO_SCENE)"
	$(PY) -m forge_viewer.cli audit "$(MYO_SCENE)" --strict

## F6 打开 Camera 面板，在 free / overview / ball_camera 之间切换；ball_camera 随自由体运动。
cameras:
	$(PY) -m forge_viewer.cli view mujoco_visuals $(ARGS)

## F9 打开 Settings，MuJoCo visual groups 的 3 号开关控制粉色 collision_debug 几何；拾取同步过滤。
geom-groups:
	$(PY) -m forge_viewer.cli view mujoco_visuals --paused $(ARGS)

## flex 1D 圆管、2D 布料、3D 软体与双骨骼 skin；F9 可切 MuJoCo visual group。
deformables:
	$(PY) -m forge_viewer.cli view deformables --paused $(ARGS)

## 列出能加载的资产（含"有没有自由体"）与本机缺的可选依赖
assets:
	$(PY) -m forge_viewer.cli assets

backends:
	$(PY) -m forge_viewer.cli backends

doctor:
	$(PY) -m forge_viewer.cli doctor $(SCENE) $(ARGS)

clean:
	rm -rf out .pytest_cache **/__pycache__
