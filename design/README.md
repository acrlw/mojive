# Mojive UI 设计稿

用浏览器打开 `index.html` 即可阅读（支持 `file://` 直接打开，无外部依赖）。
`output/ui-design-baseline/` 是本机生成的基线运行截图目录（1920×1080，菜单截图为 1600×1000 实机）。截图不提交到源码目录；运行 `tools/capture_states.py` 可重新生成。截图保留采集时的本地化状态，仅用于现状审计，不作为当前功能清单或新 UI 文案的来源。
`tools/capture_states.py` 可复现这些基线截图；`tools/render_ui_feasibility.py` 会创建项目真实的
`Window` / ImGui 上下文，标准控件直接调用 ImGui，overlay 与图标复用 `ImguiDraw2D`，最后从
OpenGL framebuffer 读取并保存 `output/ui-drawing-feasibility.png`。Pillow 只负责写出 PNG，
不参与 UI 绘制。探针显式应用 `index.html` §5 的完整色卡与 §5.9 交互状态表，
不从尚未同步的运行时 `theme.py` 继承旧 Primary 或旧控件状态颜色。

交互运行（可直接体验 hover、active、工具切换、Settings 与数值单位切换）：

```bash
make ui-feasibility
```

窗口顶部 `Probe` 菜单可在 `Workspace`、`Panels` 与 `Geometry` 三页间切换，并独立开关 Playback、
Tool column、Joint gizmos、Context hints、Value input、Joint value input、Settings、Keyframes 与 Output。
Construction 分组可整体或逐层
显示 Icon bounds、State circles 与 Geometry notes；Workspace 默认关闭三层 construction overlay，直接显示
产品态，Geometry 页的 Playback / Tools 切片仍强制显示构造层用于量距。`Panels` 页实际绘制 Control、
Joints、Camera、Inspector、Hierarchy、Assets、Stats 和 Sensors，用来检查面板宽度、长 label、
统一灰底绿勾 checkbox、滑块和表格。Control 的 equality 使用左 label / 右 checkbox 两列 property table；
Hierarchy 使用无可见表头、无内部边框的列约束，树深度只改变名称单元格内部缩进，25 px 内容行之间保留 6 px 节奏，
不使用交错行底色；展开/折叠符使用与行高匹配的自绘实心三角形，不依赖字体中的小三角 glyph。
节点名称保持白色，类型只由右侧固定列表达，不再重复绘制类型色点。名称和类型也不会依赖空格碰运气对齐。所有 property table 行显式使用一个 framed-control
高度加上下 cell padding，label 与只读值按该控件高度计算垂直中心，不依赖 cell 内容碰巧撑出行高。`Geometry` 页使用不可关闭的 `Playback`、`Tools`、`Hints & input`、
`Transform gizmos`、`Joint & helpers`、`Status`、`Shell & settings`、`Panels`、`Workspaces` 九个 ImGui tab，覆盖 M1–M18。
`Status` 单独展示 paused / running / no selection，并把选中对象名从视口标签移到持久状态栏。
Workspace 的 Output 行支持右键 `Copy message / Copy complete row / Copy all shown / Clear output`，选中行后
也可用 `Ctrl/Cmd+C` 复制整行。
Playback 中 Play 与 Pause 都会把 bounding circle、状态底圆和胶囊边界放大常显，同时保留产品态对照。
前 3 页右侧的 `Geometry controls` 可实时调整三层半径、
圆心距、分组距、分割线、预览倍率，以及单行 Context hint 的控件高度、padding 和组间距；
工具 glyph 的基准线宽可在 `1.0–2.2 px` 内调整，Rotate 交叉环的单侧遮断宽度可用
`Ring gap` 在 `0.3–1.0 px` 内调整；`Reset suggested` 恢复当前建议值。
也可直接生成各页静态验收图：

```bash
make ui-gallery
```

单页命令如下：

当前建议值采用最近一次人工调参结果：overlay 半径 `10 / 16 / 22 px`、常规圆心距 `36 px`、
工具分组圆心距 `46 px`；Context hint 控件高 `18 px`、水平 / 垂直 padding `16 / 8 px`、
组内 / 组间距 `8 / 24 px`、组合输入内部 Chord gap `10 px`、键帽水平 padding `8 px`、鼠标图标 `14×18 px`。
Rotate 图标按 ISO 正交相机投影绘制三个前半环和一个 screen ring，交叉处用背景色遮断边建立前后关系；
Snap 图标按“两条直线 + 一个精确半圆”绘制；鼠标左键填充与鼠标外壳复用同一段顶部圆角，
不会用圆点或独立圆角块近似。工具列只包含 Move、Rotate、World / Body、Snap 四项；
2D / 3D Gizmo 仅在 Settings 中切换。

```bash
.venv/bin/python design/tools/render_ui_feasibility.py --page workspace
.venv/bin/python design/tools/render_ui_feasibility.py --page panels -o output/ui-panel-feasibility.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab playback -o output/ui-geometry-playback.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab tools -o output/ui-geometry-tools.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab hints -o output/ui-geometry-hints.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab gizmos -o output/ui-geometry-transform-gizmos.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab helpers -o output/ui-geometry-joint-helpers.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab status -o output/ui-geometry-status.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab shell -o output/ui-geometry-shell.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab panels -o output/ui-geometry-panels.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab workspaces -o output/ui-geometry-workspaces.png
```

图标策略是“SVG 作为设计源文件、构建时编译成内部路径、运行时由 `Draw2D` 绘制”。
`icons/ui-icons.svg` 已放入第一组 Play / Pause / Step / Stop / Camera / Light 源图标；
`tools/compile_svg_icons.py` 只接受 circle、line、polygon、polyline、rect，并校验可见范围没有越过共同的
20×20 viewBox。可用下面的命令生成确定性的 Python 路径数据：

```bash
.venv/bin/python design/tools/compile_svg_icons.py
```

生成物写入 `output/ui_icons_generated.py`。该脚本仅在开发 / 构建阶段运行；应用启动和帧循环不解析 SVG，
也不引入光栅化库。`Joint & helpers` 页用 camera/light 的多倍率样例验证路径缩放，3D gizmo 的深度、
遮挡与 picking 则保留给真实 renderer 的 GPU 验收。

矢量路径、线宽、bounding circle、状态圆和胶囊会共同跟随 `ui_scale`。可用下面的命令直接检查
150% 缩放，不会切换到另一套位图素材：

```bash
.venv/bin/python design/tools/render_ui_feasibility.py --interactive --ui-scale 1.5
```

交互模式默认使用 vsync、关闭 framebuffer MSAA，并以 30 FPS 作为额外上限，避免 Retina framebuffer
无上限重绘拖慢输入；可用 `--fps 60` 临时提高预览刷新率。固定视角图标与 hinge arc 已预计算，生产 gizmo
验收样例也会复用固定 CameraView、ObjectGizmo 和 NumPy buffer；不可见 tab 不执行对应绘制。

关闭窗口或连续按两次 `Esc` 退出（第一次关闭数值输入窗口）；也可以在终端按 `Ctrl+C`。
页面结构：封面 / 现状切片 / 交互地图 / 动线分析 / 问题清单 / 概念设计 / 落地路线。
概念稿色板以 `index.html` §5 为准；运行时 `src/mojive/ui/theme.py` 已同步通用 Primary、surface、
checkbox 与 Inspector badge token；场景内 XYZ gizmo 的默认轴色仍按人工验收结论单独决定，不在本轮暗改。
UI 字体栈取自 `src/mojive/ui/fonts.py`：JetBrains Mono 优先，并保留项目当前的等宽回退链。

设计文档的说明文字使用中文；所有概念稿与详细原型中的产品 UI 文案统一使用英文。
英文是设计与实现的 source language，后续再按明确的 localization key 清单决定哪些文案提供中文翻译。源码文案是功能审计基线；设计稿明确标出的精简文案与动态标题是下一版 approved source copy，实现必须同步，不能为了沿用旧字符串而保留冗长 UI。新增控件尽量用图标；可从控件本身读懂的行为不再附加说明句。

逐模块落地顺序、自绘边界、布局持久化和验收矩阵见
`plan/UI_REFACTOR_IMPLEMENTATION.md`。
