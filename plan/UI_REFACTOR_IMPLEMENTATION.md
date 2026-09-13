# Mojive UI 交互重构落地记录

状态：历史实现记录。M1–M18、D1–D8 已落地；当前 UI 行为以代码、用户指南和 `make ui-gallery`
产物为准。本文保留当时的设计约束和阶段记录，不作为现行操作手册。
概念设计来源：`design/index.html` 的 M1–M18 与 D1–D8
绘制可行性探针：`python/tools/ui_feasibility.py`

## 1. 目标

这次重构要把概念设计完整地转化为一套连贯的 ImGui UI，而不是零散调整几个控件。
对于尚未完全认同的概念，也先保留为明确的实施与验收项，不能仅因为当前代码更容易维持原状就将其删除。
最终是否保留，由实际运行后的视觉和操作验收决定。

完成后的 UI 应满足以下要求：

- 所有产品 UI 使用专业、统一的英文源码文案；后续中文翻译必须从这些确定的英文 source string 映射，
  不再出现一部分翻译、一部分不翻译的混杂状态。
- 沿用 JetBrains Mono 字体栈和现有 CJK fallback，不在本轮重新设计字体加载架构。
- 播放胶囊仅保留 Play/Pause、Step、Stop；本轮 UI 不提供仿真速率控件。
- Control 同时负责 actuator 与 equality constraint；Joints 只查看和控制 joint。
- Settings 改为非模态、可停靠面板，并提供可恢复的默认工作区。
- 落地紧凑的视口 overlay、数值输入 popover、joint 标签、Output 行选择和 Keyframes 工具栏。
- 标准 ImGui 控件与自绘控件共享同一套颜色、间距和交互状态规范。
- OpenGL、WGPU、1× 和 HiDPI 下的布局与交互一致。

除非某项新交互缺少必要命令，例如点击 joint limit 标签将 joint 设置到端点，否则本轮不改变 scene、adapter
或 command 的业务语义。

## 2. 当前代码审计结论

概念稿中的大部分结构可以复用现有实现，不需要推倒重来：

- `ui/window.py` 使用 DockBuilder 建立左侧 22%、右侧 30%、底部 26% 的默认布局；Reset Layout 会立即
  重建并持久化这一布局。
- 交互式 viewer 把 `imgui.ini` 写到用户配置目录，避免污染项目工作树；headless 模式关闭 ini 持久化。
- 面板可见标题已经使用稳定的 `###EnglishName` 作为内部 ID。这能避免语言切换破坏布局，应继续保持。
- `ImguiDraw2D` 已提供线段、折线、矩形、圆形、多边形、文字和图像等概念稿所需原语。
- 播放控件已采用 `InvisibleButton + draw list`，Play 与 Pause 均使用一致的圆形状态几何。
- joint range gizmo 和 Keyframes dope sheet 已经是自绘实现，应在现有基础上重排和统一样式，
  不应重复开发新的简化版本。
- Output 已改成 clipped selectable 行与右键复制；Control/Joints 职责已拆分；Settings 已改成可停靠的
  非模态面板。
- 运行时 `ui/theme.py` 与设计稿现已统一使用主色 `#9CBF8D / #B8D2AC / #67875A`；轴色、joint 色与
  Inspector badge 继续按各自语义 token 管理，不能退回散落的局部常量。
- 当前 `ControlPanel` 的 `speed` 是 Mojive 自己已有的控制，不是 playback overlay 的要求。MuJoCo
  3.11 viewer 也存在 `PERCENT_REALTIME` / `real_time_index` slowdown 档位，但本轮产品方案明确不暴露该控件；
  删除 UI 入口时需要同时决定是否保留命令/API 能力，不能用“MuJoCo 有”反推面板必须显示。
- 当前 Inspector Transform 使用彩色 X/Y/Z reset button 与紧邻的 `drag_float` 复合字段；基础行是
  `position` 与 `rotation`，选中 free body 时还在同一 `bg_popup` child 内显示只读 `linear velocity` /
  `angular velocity`，velocity label 单独占一行、下一行放三组轴字段。最新实现还使用紧贴折叠标题的
  `bg_popup` child、12 px 字体和 3 px 轴组间距。概念探针必须逐项复刻，不得添加 `scale`，也不得改成
  一个三值文本框。每一组轴字段必须把彩色轴 button 与 drag field 画成一个连续圆角矩形：仅外侧两端
  保留圆角，内部 seam 填平，不能保留两个相邻控件各自的内侧圆角。
- 2D transform gizmo 的默认色来自 RGB axis，任意轴或环 hover/active 统一使用色卡 Active
  （Primary Bright `#B8D2AC`）。运行时与可行性页都使用这一目标态；3D solid 形态继续由
  OpenGL/WebGPU renderer 验收，不能用 2D draw list 假装 3D。
- joint range 目标态使用紫色主轴/圆弧、MIN 蓝 tick、MAX 红 tick、Primary Bright 当前值 tick，以及带
  底板和语义色点的白字标签。Hinge 不绘制方向箭头；slide 只保留一支与 transform gizmo 相同轮廓的
  轴外箭头，且只有该箭头可 hover / press / drag，范围线、当前 tick 与 MIN/MAX 标签均为只读语境。
- transform rotation 与 hinge joint rotation 的拖动反馈共用新色卡：sweep sector 使用 Primary Dim
  `#67875A` 24% alpha，活动弧和当前 / 命中 snap tick 使用 Primary Bright `#B8D2AC`，普通 Shift snap
  ticks 使用 Text Disabled；旧黄色 / 橙色只作为历史审计色，不进入目标态。

## 3. 标准 ImGui 与自绘边界

### 3.1 使用标准 ImGui

凡是键盘导航、文本编辑、裁剪、列表选择和 docking 比特殊外观更重要的部分，优先使用标准 ImGui：

- menu、dockspace、tab、table、tree row、list selection、context menu 和 tooltip；
- Settings 参数行、普通输入框、combo、slider 和 disabled 状态；
- Control、Joints、Camera、Hierarchy、Assets、Stats、Sensors、Output 的面板结构；
- 数值输入 popover 内的输入框以及 Relative/Absolute combo；
- Output 消息行选择、右键菜单、`Ctrl+C` 和滚动。

### 3.2 使用 draw list 自绘

标准控件不能准确表达以下视觉时，使用 `ImguiDraw2D` 或轻量图标 helper：

- 播放胶囊和视口工具列中的圆形底色、线条图标和多边形图标；
- 左键双击的鼠标轮廓、左键高亮和 `×2`；
- Keyframes 的紧凑 transport 图标；
- joint MIN/MAX 标签底板、语义色点、端点刻度和点击区域；
- screen-constant camera/light helper 图标，以及选中后的 influence volume 线框；
- gizmo 专用的箭头、圆弧、handle 和 active 状态几何。

### 3.3 混合控件

以下组件使用标准或 invisible ImGui hit target，再在上层绘制设计稿视觉：

- 圆形图标按钮和 segmented control；
- 单行情境手势提示条；
- 数值输入 popover 外壳与 degree/radian 选择器；
- Keyframes command bar 和已有 dope sheet；
- hinge 的可点击 joint limit 标签；slide limit 标签保持只读。

混合控件的可见图形可以是圆形，但键盘导航和鼠标 hit target 至少为 28 个逻辑像素的矩形区域。
文字名称和快捷键放入 tooltip。所有尺寸先按逻辑像素定义，再乘以 `style_scale`；一像素线条最终对齐到
framebuffer 像素网格，避免 HiDPI 下发虚。

### 3.4 SVG 的使用边界

结论：**SVG 适合作为图标的设计源文件，但不作为 Mojive 的运行时绘制格式。**

Dear ImGui 负责提交 draw list 或已经上传的纹理，不负责解析 SVG、曲线细分、光栅化和 GPU 上传。
当前项目也没有 SVG 解析器；若运行时直接加载 SVG，需要同时增加解析 / tessellation、缓存失效、
OpenGL 与 WGPU 两套纹理上传和 HiDPI 重建路径，收益不足以抵消复杂度。

建议采用以下离线流程：

1. 在 `design/icons/` 保存规范化 SVG 源文件，统一 `viewBox="-10 -10 20 20"`，只使用无滤镜、无文字、
   无 mask 的单色 path / line / polygon / circle；颜色使用 `currentColor`，不把主题色烘焙进文件。
2. 开发期脚本将 SVG 转成确定性的内部路径常量，例如 `ui/icons_generated.py`；生成结果随源码提交，
   应用启动和帧循环不读取 SVG，也不引入新的运行时依赖。
3. 生成器校验所有可见 stroke 外缘都位于 `r=10` bounding circle 内，并输出稳定顺序，便于 review diff。
4. 运行时仍通过 `Draw2D` 绘制，可按 `ui_scale` 缩放、按状态换色，并继续兼容 OpenGL / WGPU。
5. 只有普通 2D UI 图标和 camera/light 的屏幕空间 helper 使用这条路径；3D gizmo 的深度、遮挡、
   picking 与场景比例仍由 renderer 中的 mesh / line geometry 实现，不能用 SVG 纹理替代。

当前可行性实现已提供 `design/icons/ui-icons.svg` 与 `design/tools/compile_svg_icons.py`。第一批只包含
Play / Pause / Step / Stop / Camera / Light，编译结果写入 `output/ui_icons_generated.py`；Move / Rotate /
World–Body 等需要状态化参数或真实投影的图形暂不强行 SVG 化。生成器只依赖 Python 标准库，拒绝 path、
filter、mask、文字和越界 stroke，使未来接入 runtime 时不需要 SVG parser 或新增动态库。

如果未来图标数量增长到数百个且路径提交成为实测瓶颈，再评估构建时生成单色 SDF / MSDF atlas；
当前没有证据需要提前承担 atlas、纹理生命周期和多倍率缓存的复杂度。

## 4. 自绘组件结构

不要继续把图标和绘制逻辑散落在各个 panel 中。建议补充以下小型共用模块：

- `ui/icons.py`：统一的 20×20 逻辑坐标图标，只由 line、polygon、rect、circle 与参数化圆弧组成；
- `ui/widgets.py`：`icon_button`、`circular_icon_button`、`segmented_control`、
  `property_table` 和统一 checkbox 状态；
- `ui/overlays.py`：viewport 锚点、保留区域、碰撞避让和边缘 clamping；
- `ui/labels.py`：文字测量、标签底板、语义色点、hover/pressed hit region。

具体绘制规则如下。

### 播放胶囊

- 所有矢量 glyph 在共同的 20×20 逻辑坐标系内创作，顶点和 stroke 外缘必须被 `r=10 px` bounding circle 完整包住；
- 图标包围圆、状态底圆、胶囊半高统一为 `r=10 / 16 / 22 px`，间距严格等差 6 px；
- 3 个命中单元均为 36×36 px，圆心距 36 px，状态圆间净空 4 px，完整胶囊为 116×44 px；
- Paused 状态第一键绘制 Play 且保持 neutral；Playing 状态同一键切换为 Pause，并保持 playing selected 的 32 px 正圆底色；视觉验收切片必须并排展示两态；3× construction view 必须分别绘制 Play 与 Pause，并让两者都同时显示 r=10 bounding circle、r=16 state circle 与 r=22 胶囊端部，直接核对两段 6 px 径向间距；
- 默认态不画底圆，只有 hover / pressed / playing selected 画 32 px 正圆；
- DPI 缩放只对路径坐标、线宽和三层半径统一乘 `ui_scale`，禁止按单个 glyph 追加经验倍率。

### 左侧工具列

- 与播放胶囊共用 `r=10 / 16 / 22 px` 三层半径与 36×36 px 常规命中单元；
- 常规圆心距 36 px，分组间圆心距 46 px，所有图标共享同一尺寸的隐形包围圆；
- 四个工具 glyph 共用 20×20 逻辑坐标和 1.42 px 基准线宽；Rotate 的视角固定为 yaw −135° / pitch 30°，X/Y/Z 三个前半环必须使用离线预计算、常量化的 r=5.8 二维路径，绘制热路径只做 DPI 缩放和平移，禁止逐帧执行投影、三角函数或曲线重采样；screen ring 以可见外缘 r=10 为不变量，根据线宽反推路径半径（默认可见线宽 1.136 px、中心线 r=9.432），使图标始终填满共同视觉包围；四环统一白色，每条环先用 surface color 画宽线再覆白线，默认形成 0.32 px 单侧透明遮断边，后绘环在交叉处遮住前绘环；内环拟合必须把遮断边计入包围尺寸，并与 screen ring 保留可见负空间；World–Body 使用等长正交投影轴、箭头和带空白遮断圈的中心点；Snap 由两条直线与一个精确半圆组成，圆弧按最终显示半径增加采样，不使用折点近似；
- Move 使用两条 1.648 px 实心杆与四个箭头组成连续轮廓，杆端延伸至箭头底部内侧 1.2 px；World–Body 的三组固定屏幕方向共用同一个七点局部箭头轮廓，再分别旋转到 −90° / 30° / 150°，因此三角头和中心轴严格共线；轴杆宽度与 Rotate 可见环宽统一为 `Tool stroke × 0.8`（默认均为 1.136 px），调节 Tool stroke 时同步变化；最后用 surface color 覆盖 r=2.4 中心遮罩并绘制 r=1.2 圆点；右上空白扇区使用当前 UI 等宽字体的 `W` / `B` 表示当前 world frame / body frame，使用 glyph ink box 等比缩放并居中到 x=3.0…7.4、y=−6.4…−2.4 的无底板安全区，禁止使用折线手绘字母，任何字符不得遮挡坐标轴或越过 r=10 bounding circle；点击图标或按 `T` 切换，tooltip 随当前状态使用 `World frame (T)` / `Body frame (T)`；禁止把杆身和三角头作为两个独立抗锯齿图元，避免斜轴产生视觉偏移；预计算点集和 DPI 缩放后的临时顶点缓存均复用，不在帧循环中创建可变长度容器；
- Move / Rotate / World–Body 与 Snap 之间使用 20 px、一像素、72% Border alpha 的短分隔线；
- 2D / 3D Gizmo 是低频显示偏好，只保留 Settings → Interaction → Gizmo → Style，不在工具列提供重复图标；
- active 状态使用 `BG_FRAME_ACTIVE` 底和 `PRIMARY_BRIGHT` 图标。

### M6 情境提示条

- 提示条按输入状态互斥切换：无选择时显示常用视口导航 `[left mouse] Orbit · [right mouse] Pan · [wheel mouse] Zoom · [F] Frame`；transform ready 显示 `Snap (Shift) · [T] World / Body · [mouse ×2] Type value · [Ctrl] + Drag · [left mouse] Push · [right mouse] Twist`；transform dragging 只显示 `Snap (Shift)`；Ctrl held 只保留同一组物理扰动输入。ViewCube 不进入常驻提示条，只在 hover 时用 tooltip 标出当前面或动作。数值输入 popover 或任意 blocking popup 打开时隐藏提示条。完整替代手势（middle drag、Shift + left drag 等）仍保留在 Help，不把整张快捷键表塞进情境提示条；
- 不得显示 `[Ctrl] Fine` 或 gizmo 状态下的 `[Esc] Cancel`：当前产品没有 Ctrl 微调，也没有 Esc 取消 gizmo 拖动；Ctrl 已保留给物理扰动，Esc 只取消数值输入 popover；
- 整个容器必须由上下直线和左右两个精确半圆构成；ImGui 实现使用预计算半圆点集生成凸轮廓，不得把 `rounding = height / 2` 交给 rounded-rect 内部钳制，也不得用小圆角矩形代替胶囊；
- 每行 kbd、鼠标图标、次数和说明文字共享同一垂直中心；
- Snap 使用动作优先的 `Snap (Shift)`；其余项目使用“左侧输入方式、右侧功能 label”：`[T] World / Body`、`[mouse ×2] Type value`；物理扰动使用共享修饰键的 chord group：`[Ctrl] + Drag · [left mouse] Push · [right mouse] Twist`，不显示脱离鼠标输入的孤立 `[Ctrl] Perturb`，也不在左右鼠标图标后重复两次 `drag`；
- 普通 hint group 间距默认 24 px；`Ctrl + Drag + mouse` 内部使用独立、可调的 `Chord gap`，默认 10 px，适用范围 4–20 px，不得复用普通 group gap；
- 键帽使用对称 8 px 水平 padding，文字按实际测量宽高在背景框中双向居中；
- 鼠标图标为 14×18 px，与 18 px 键帽同高，不得单独撑高行布局；
- 胶囊高度由 `control_height + 2 × padding_y` 推导，默认 `18 + 2 × 8 = 34 px`，禁止硬编码一个小于内容高度的容器；
- 胶囊宽度由每个输入、label、组内间距、组间距和左右 padding 的实测总和推导，不允许使用固定宽度裁切；
- 不写 `Double-click` 或 `LMB`；
- 画出鼠标轮廓，将完整左键区域填色，并在旁边显示 `×2`；填充与外壳必须共享顶部圆角轮廓，禁止叠加圆点或独立圆角矩形；
- 只显示当前操作需要的短动作，不复制 Help 中的完整说明。

### Status

- Status 是所有 workspace 持久可见的底部通栏，不隐藏在 Settings 或 Stats 中；可行性程序提供独立
  `Status` tab 展示 paused / running / no selection 三态；
- 当前选中对象名只在 Status 左侧显示，视口不再重复绘制对象名标签；无选择时显示 `no selection`；
- running 状态使用 12% Warning 淡底、2px Warning 顶边与 Warning 状态分段，并补充 sim time；对象名和 renderer/FPS 保持白字，详细的帧时间、物理和渲染历史仍归 Stats；
- 视口底板标签只用于 transform 拖动值、joint MIN/MAX、相机预览和必要调试值。

### 数值输入 popover

- 目标宽度 220 个逻辑像素；
- 标题直接显示动作与对象，例如 `Rotate Z`、`Move slide_joint`；
- 不显示 `Precise Input`、Apply、Cancel；
- Enter 提交并退出，Escape 取消并退出，点击外部取消；
- rotation 显示 `° / rad` segmented control；translation 只显示当前源码单位；
- 只有确实支持 Relative/Absolute 时才显示 Mode；
- 超长动态名称中间省略，tooltip 显示完整名称，popover 本身不随名称扩宽。

### Joint limit gizmo

- 主轴和圆弧使用紫色，暂不增加 halo 或白色外描边；
- MIN tick 使用 Z 轴蓝 `#6F94E5`；
- MAX tick 使用 X 轴红 `#DC7773`；
- 主轴 / 圆弧使用正式 Joint token `#AF84B7`，不再使用 runtime `#B87AF2` 或 feasibility
  临时值 `#9B6BD3`；
- slide / hinge 的当前位置分别使用 20 px × 4 px 的垂直 / 径向 Primary Bright tick；hinge 方向只由紫色
  圆弧表达，slide 另保留与 transform axis 相同的七点箭头作为易命中的拖拽 affordance；
- 标签使用不透明深色底板和白色文字，颜色只放在 6 px 语义圆点上；
- 通过 `CalcTextSize` 计算标签宽度，保证数值与单位不会重叠；
- hover 使用 `BG_FRAME_HOVERED + PRIMARY_BRIGHT`；press 使用 `BG_FRAME_ACTIVE`；
- hinge 点击标签提交明确的 set-to-limit command，tooltip 分别为 `Set minimum` 和 `Set maximum`；slide
  标签不创建 InvisibleButton，也不改变 hover / pressed 状态。
- 双击 joint 轴线或圆弧打开以关节名为标题的 Type value popover；slide 固定单位 `m`，hinge 复用
  degree / radian 单位逻辑，Enter 提交、Esc 取消。

### 输入路由、快捷键与 overlay 扩展

- viewport 输入每帧严格按 `blocking modal / popup → UI capture → GestureRouter → gizmo / camera / perturb`
  顺序归属。blocking prompt 出现时立即 abort 已持有的 gesture，隐藏情境提示，并禁止 playback、tool
  column、camera、gizmo、pick 与 perturb 接收新的键鼠操作；不能只依赖 ImGui 的视觉 dimming。
- `ui/input_bindings.py` 是动作到按键和显示 label 的唯一来源；轮询、tooltip 与情境提示使用同一个
  `InputBindings` 实例。未来 key-mapping 设置页只替换 binding map，不修改 gesture 判定或 UI 文案。
- playback、tool column 与 hint 使用 declarative control/group 描述生成尺寸、圆心、分隔线、绘制和
  tooltip；新增工具或新增分组只增加 descriptor，不复制定位公式。
- 运行时 hint 在共享逻辑几何之上使用 0.76 的独立视觉倍率，以低于 playback/tool column 的注意力层级
  呈现；命中区域只属于交互控件，hint 本身始终 `no_inputs`。
- 1600×1000、vsync off 的 production 采样为 6.36 ms/frame（约 157 fps）；去掉三个 capsule overlay 后
  差异小于 0.05 ms/frame，落在测量噪声内。profile 保存于 `output/ui-frame-profile.prof`，主要累计热点为
  swap/render、gizmo 投影和 Hierarchy 行绘制。camera/light 固定 glyph 路径已改为模块级预计算，不在帧内
  重建角度与二维路径数组。

### Settings

- 所有参数行使用两列表格，默认宽度比例 36/64；
- 左侧 label 右对齐，右侧控件左对齐；
- 通过分组标题吸收重复上下文，使固定 label 尽量保持一到两个词；
- 固定设置名不能依赖省略号才能理解；
- 动态 domain name 可以省略，但必须提供完整 tooltip。

### Camera

- `yaw`、`pitch`、`distance`、`fov_y_deg`、`far` 和 `projection` 必须提交到真实的两列 ImGui table；
- 左列 label 右对齐、右列 widget 拉伸，禁止继续依靠 slider 的尾随 label 做自由混排；
- presets 使用四列两行 table：`front / back / left / right` 与 `top / bottom / iso / frame all`；`frame all`
  不单独落到第三行；
- projection 使用 `persp / ortho` 互斥 segmented control，不使用单独的 `orthographic` checkbox；
- source、presets 与 camera bookmarks 保持各自分组，不塞进参数 table。

### Hierarchy

- 树行使用 node / type / visibility 三列 ImGui table；node 弹性伸缩，type 与 visibility 固定宽度；
- 展开符、白色名称、type 和可见性图标共享同一垂直中心，selected 背景跨整行；类型只在右侧列显示，
  不再用颜色圆点重复编码；
- 不在 full-width selectable 后通过 `same_line` 追放 type，以免基线和行高漂移。

### Output

- 整条消息行 selectable；
- 右键菜单提供 Copy 和 Copy All；
- `Ctrl+C` 复制选中消息，`Ctrl+Shift+C` 复制全部；
- 每条消息不再创建 Copy 按钮；
- 大量消息时使用 clipping，避免行数增加导致控件数量与绘制成本同步爆炸。

### Keyframes

- 顶部按语义分成 Take transport 与 Snapshot 两条紧凑 command row，不再堆两排长文字按钮；
- 图标本体 16 px，hit target 28 px；
- `Record New Take` 与 `Capture Snapshot` 保留短文字主操作，其余 seek / replay / stop / clear / view 命令使用图标；Record 是唯一使用语义色的主要动作；
- 文字名称和快捷键放在 tooltip 中；
- 保留现有 dope sheet、recorded take 和 selected-key 编辑能力，只调整入口层级和布局。

绘制可行性图可使用以下命令复现：

```bash
.venv/bin/python -m mojive.tools.ui_feasibility
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab playback
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab tools
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab hints
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab gizmos
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab helpers
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab status
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab shell
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab panels
.venv/bin/python -m mojive.tools.ui_feasibility --page geometry --geometry-tab workspaces
```

该探针创建项目真实的 `Window` / ImGui 上下文：标准控件由 ImGui 渲染，自绘图形复用
`ImguiDraw2D`，并从 OpenGL framebuffer 读取结果。它可以验证字体栅格化、主题、DPI、裁剪、
控件状态、图形原语和间距，但仍不代替带真实 scene 与完整交互状态的最终 GPU 截图验收。
交互模式关闭 4× framebuffer MSAA，启用 vsync 并再以 30 FPS 软件上限兜底；这不是渲染 benchmark，
不能在 Retina framebuffer 上无上限重绘。固定视角图标和 M8 hinge arc 使用预计算常量，帧内只做缩放和平移。
Geometry 页使用不可关闭的 `Playback / Tools / Hints & input / Transform gizmos / Joint & helpers / Status /
Shell & settings / Panels / Workspaces` 九个 ImGui tab；前 3 页共享 Geometry controls，其余页使用完整内容宽度。
页签覆盖 M1–M18，并避免为了塞进单页而压缩图例、面板与时间工作区。

## 5. 主题与英文文案规范

在调整 panel 之前先同步 `ui/theme.py`，避免标准控件和自绘控件继续使用两套颜色：

- Primary：`#9CBF8D`；Primary Bright：`#B8D2AC`；Primary Dim：`#67875A`；
- 空间功能色：Axis X `#DC7773`、Axis Y `#52AA5C`、Axis Z `#6F94E5`、Joint `#AF84B7`；
  X/Y/Z 都为细线可读性保留较高色度；Y 采用偏冷的功能绿，与 Primary 拉开；Joint 使用偏暖、
  较低色度的紫，与 Z 蓝拉开；
- Inspector 的 X/Y/Z 徽章底不直接复用亮 Axis 色，而使用同色相深色 surface：
  `#A84E4B / #317A3A / #4868AD`；徽章文字固定纯白 `#FFFFFF`，数值输入区继续使用
  `BG_FRAME + Text #DCDEE3`；overlay 标签只在 6 px 圆点使用 Axis 色，文字本体仍为 Text；
- frame default/hover/active：`#2B2F34 / #363B41 / #40464D`；
- selected row 使用 `BG_HEADER`，hovered row 使用 `BG_FRAME_HOVERED`；
- checkbox 使用灰色 frame 状态和 `PRIMARY_BRIGHT` check mark，不使用蓝色整块填充；
- feasibility probe 使用统一 checkbox helper 明确绘制 `BG_FRAME / BG_FRAME_HOVERED /
  BG_FRAME_ACTIVE + PRIMARY_BRIGHT check`，避免平台或绑定层 checked 颜色重新落回默认蓝；
- selected tab 只改变底色，移除顶部 overline；
- 红蓝只承担 axis/limit 语义，不用作普通文字高亮。

UI 文案必须复用仓库已经确定的英文术语，不能为了“显得更专业”自行替换近义词。
先确定最终英文 source string，再加入 localization 映射。面板内部 ID 始终保留稳定英文后缀，例如：

```python
translated_title + "###Settings"
```

这样切换语言不会破坏已保存的 docking 布局。

## 6. `imgui.ini` 与布局持久化

### 结论

不把运行时生成的 `imgui.ini` 提交到 Git。

该文件包含机器相关的窗口坐标、viewport 信息和用户临时布局。直接跟踪会造成频繁 diff，
也可能把开发机布局错误地变成所有用户的默认布局。仓库目前忽略它是合理的。

### 建议方案

1. 在 `settings_path()` 附近提取共用的 `config_dir()`。
2. 运行时布局保存到 `<config-dir>/mojive/layout.ini`。
3. 支持 `MOJIVE_IMGUI_INI` 环境变量，供测试和明确覆盖使用。
4. headless 和普通 GPU 测试默认继续使用空 ini path；只有布局持久化测试传入临时文件。
5. 默认布局不保存成仓库内的 ini blob，而是在 Python 中声明：
   - layout schema version；
   - 稳定 panel ID；
   - panel 所属 dock group；
   - left/right/bottom split ratio。
6. 首次启动、layout 文件损坏或根 dock node 丢失时，使用 DockBuilder 创建默认布局并立即保存。
7. `Window > Reset Layout` 删除当前 dock tree，重新建立代码内默认布局并立即保存；
   不能影响 Settings 内容、模型数据或其他用户配置。
8. schema 升级不主动覆盖用户布局：
   - 新增 panel 只在没有保存 docking assignment 时放入默认分组；
   - panel 改名必须提供旧 ID 到新 ID 的明确迁移；
   - 只有布局损坏、缺失或用户主动 Reset Layout 时才完整重建。
9. OpenGL 和 WGPU 共用同一套 ini path、迁移与 reset 逻辑，删除目前重复设置 ini filename 的分支代码。

重构后的默认分组：

- 左侧 22%：Hierarchy、Assets、Inspector；
- 右侧 30%：Control、Joints、Camera、Settings、Sensors；
- 底部 26%：Stats、Output、Keyframes、Plot、Help、Info；
- 中央：Viewport。

Settings 默认停靠在右侧，但用户可以像其他 panel 一样拖出、移动或重新停靠。
因此概念稿中 Settings 的其他非模态摆放方式不需要分别开发。

## 7. 分阶段实施方案

每个阶段都应形成可单独审查、可回退的提交，不与无关业务改动混在一起。

### 阶段 0：UI 基础设施

对应后续所有 M/D 模块的共同依赖。

- 同步主题 token 与控件状态；
- 增加共享 icons、hybrid widgets、label measurement 和 overlay layout helper；
- 将 ini 保存迁移到用户配置目录；
- 增加代码声明的默认布局、非破坏迁移和 Reset Layout；
- 增加 geometry、theme token、stable panel ID、layout path 单元测试。

验收：theme state gallery、首次启动布局、重启保持布局、Reset Layout、OpenGL/WGPU 一致性。

### 阶段 1：应用外壳

对应 M1、M2、M4–M7，以及 D1、D2、D4。

- 完成 File/Edit/Entity/View/Window/Help 的菜单职责，capture/record 收进 View；
- 增加底部通栏 status bar，统一 running/paused 状态与门控文案；
- 落地圆形播放按钮、左侧工具胶囊和单行情境提示条；
- 将 Settings 改成非模态、可停靠 panel；
- 所有 viewport overlay 接入保留区域、碰撞避让和边缘 clamping。

视觉验收点：

- M6 提示条是否始终显示，还是在熟悉后淡出；
- Settings 默认是否应停靠右侧；
- empty、normal、running、perturbation 四种状态下 overlay 是否过密。

在完成实际运行验收前保留概念稿行为，不提前删掉这些设计。

### 阶段 2：视口编辑

对应 M8、M10，以及 D3、D5、D8。

- 将 2D / 3D Gizmo 收敛到 Settings 的唯一入口，不在左侧工具列重复提供；
- 实现动态标题的数值输入 popover 和 `° / rad` 选择；
- 调整 joint range 图形：hinge 保留可点击 MIN/MAX，slide 只允许单箭头拖动；
- 绘制 camera/light helper 图标和选中后的 influence volume；
- 检查标签与 playback、ViewCube、camera preview 和 viewport 边缘的避让。

视觉验收点：helper 默认密度、MIN/MAX 点击方式，以及单位选择是否记忆上次状态。

### 阶段 3：Inspector 与核心面板

对应 M3、M9、M11–M14，以及 D6、D7。

- 在不改变 adapter 所有权的前提下实现 Inspector 分组与搜索；
- 使用统一 property table 重构 Settings 四页，并缩短英文 label；
- 将 actuator 从 Joints 移入 Control；第二组标题缩短为 `equality`，并用无可见表头的 ImGui
  property table 定义左 label / 右 checkbox 两列；删除 Control 的 speed、状态 KV 与 last message；
- 从 Joints 删除 actuator tab 与 `Pose · qpos` 文案；
- Camera、Hierarchy 使用统一的属性表；Hierarchy 树行最小 25 px，另留 6 px item spacing，不使用
  zebra row background 或内部行分隔线，展开/折叠符按行高自绘为 10 px 实心三角形；节点名称保持白色，
  类型只在右侧固定列显示；
- property table 统一以 `GetFrameHeight()` 加上下 cell padding 定义行高，label、只读文本和自绘 checkbox
  都必须按同一 framed-control 视觉中心垂直对齐，禁止只靠 cell 内容自动撑高；
- M14 默认使用右侧 30% 宽度，不再按旧窄面板设计。

验收：使用最长的 model、actuator、joint 和 camera 名称，在最小支持面板宽度下检查。
固定 label 不允许换行；动态名称允许省略并显示 tooltip。

### 阶段 4：辅助面板

对应 M15–M18。

- Keyframes 重排为单行 icon command bar，保留现有 dope sheet；
- Assets、Stats、Sensors 使用相同的分组与 property-row 规则；
- Output 改为 selectable row、context copy 和 keyboard copy；
- Plot、Help、Info 中适用的 selection、hover、checkbox 和 title 状态同步到新规范。

验收：

- Keyframes 在最小底部 dock 高度下仍能完成 record、replay、snapshot 和 selected-key edit；
- Output 在数百条消息与超长路径下不产生按钮爆炸、严重分配或横向布局失控。

### 阶段 5：集成与清理

- 新交互通过验收后，删除旧 modal、重复 toolbar 和过时 panel 路径；
- 检查 overlay 和 log rendering 的 transient allocation；
- 补齐 keyboard navigation、tooltip、focus restoration 和屏幕边缘 clamping；
- 根据最终英文 source copy 更新用户文档与 localization inventory。

## 8. 测试与视觉验收

新增统一入口 `make ui-gallery`，生成以下确定性截图：

- D1–D8 全窗口状态；
- playback default/hover/pressed/paused、tool pill、单行提示条；
- Settings default/hover/pressed/checked/disabled；
- position 与 rotation 数值输入，以及 degree/radian 两种单位；
- slide 单箭头 default/hover/pressed 与只读刻度，hinge joint limit 的 MIN/MAX hover/pressed；
- Control/Joints 新职责、Settings 长 label、Hierarchy 长动态名称；
- Keyframes idle/record/replay/selected-key；
- Output empty/selected/context-menu/long-path/high-message-count；
- OpenGL/WGPU 的 1× 与 2× 输出。

实施过程中先运行最小相关测试。每个阶段完成前至少运行：

```bash
make check
make gpu
make settings
make gizmo
make joint-gizmo
make gizmo-gallery
make hidpi-gallery
make ui-gallery
```

颜色和控件状态使用像素抽样；间距与 hit target 使用几何断言；整体组合、裁剪和跨 backend 一致性使用
golden image。所有生成结果继续放在 `output/`。

## 9. 需要通过实际运行决定的设计

以下内容不是遗漏，而是明确保留的实际验收点：

- Settings 初始停靠在右侧，还是首次打开为独立非模态窗口；
- M6 提示条是否在每次 manipulation 中持续显示，还是使用一段时间后淡出；
- 未选中 camera/light helper 的默认显示密度；
- 点击 MIN/MAX 是否立即移动 joint，还是先选中标签再确认；
- Keyframes 在最小 dock 高度下最合理的图标分组。

每项都通过小型组件边界实现，不创建大范围 feature flag，也不维护两套 panel。
首次运行版本按概念稿实现，再依据视觉验收确定最终默认值。

## 10. 完成标准

满足以下条件后，才能认为本轮重构完成：

- 运行时可以复现 D1–D8 的状态；
- 所有可见产品文案为英文，并与仓库术语一致；
- 自绘控件与标准 ImGui 控件具有一致的 hover、pressed、selected、disabled 和键盘行为；
- 用户布局可以跨启动保存，无需提交个人 `imgui.ini`；
- Reset Layout 能确定性恢复默认布局；
- OpenGL 与 WGPU 的完整测试和视觉验收全部通过。
