# 原生渲染完整性与 parity 验收

> 本文记录前阶段 b50e349 的验收和性能结果；当前 macOS／Linux 范围及后续资源、读回和 Metal 修正见[原生资源与 runtime 验收](native-resource-runtime.zh.md)。历史数字和当时的平台限制保留供对照，不代表当前分支的最终验收。

本轮继续在 `codex/cpp-foundation` 完成原生同步读回、可见性剔除和窗口呈现优化，并重新检查历史 UI／交互反馈。已同步 `origin/main` 的 `b4887d7`（合并提交 `c2260c8`）；可以用 `make native-viewer` 或 `make native-editor` 启动，Python API 保持兼容。

最终同步 Renderer 矩阵的 63 组成对测例中，bgfx 全部快于 OpenGL；RGB／深度／分割的耗时中位比分别为 0.523／0.543／0.481。功能、模型语料及连续运动对照通过。窗口吞吐已改善，但大视口呈现的 P95 仍高于 OpenGL，模型重编译仍可能产生几十毫秒长帧，4096 份原始 G1 几何仍不是实时工作负载。因此不能宣称所有路径都更快或所有设备都已验收。Linux、Windows 和真实 120 Hz 呈现仍需要对应设备；未启动 CI、未合并 main。

## 本轮修复的具体问题

| 用户可见问题 | 原因和修复 |
|---|---|
| 近景方块侧面条纹、平移闪烁，球面出现斜切暗区 | 阴影 shader 手动读取矩阵下标，经 HLSL／Metal 编译后取错深度轴，导致背光判断和 slope bias 错误。改为基向量乘矩阵，统一数学含义，并检查编译后的 Metal shader。 |
| 实际 Viewer 与离屏对照的构图、形状不同 | Viewer 的相机默认 aspect 为 1，原生 backend 没有按目标尺寸修正。现在由目标统一计算投影，resize 同步更新，并新增真实 Viewer 宽／窄／恢复对照。 |
| 空背景颜色不一致 | C++ 默认背景及 8-bit clear 转换与 OpenGL 不同；统一背景和舍入。材质、光照和色彩空间继续通过分对象对照检查。 |
| 开了 VSync 仍可能撕裂，拖动排队感明显 | 原实现只在 CPU sleep，未启用 GPU 呈现同步。现在 VSync 直接传递到 GPU；删除软件刷新率缓存和 sleep，删除第二层 CPU 帧队列，GPU 最多两帧在途；Metal 同步呈现保留两个 drawable，关闭 VSync 时提供第三张备用图像，缓解合成器占用造成的等待。 |
| 部分 Menagerie 文件因采样数加载失败 | 模型可请求 24 samples。Python 后端把颜色采样请求向下取可移植的 1×／2×／4×，不再因非标准或过大的请求退出。 |
| 关闭 MSAA 开关仍然抗锯齿 | Metal 按 attachment 的采样数工作，单改 draw state 不够。切换时替换颜色目标，保持尺寸、恢复行为和已提交读回的有效性。 |
| 全透明碰撞几何在分割图中消失 | 颜色可见性判断误用于语义输出。明确请求透明身份时包含 alpha=0 几何，颜色仍保持不可见。 |
| skin 场景经过组合加载失败 | MuJoCo attach 没有同步命名空间内的 skin 材质引用；仅修正新加入的 skin 名称／引用，并验证同场景重复加入。 |
| 100 humanoid 的缺失几何和重复上传 | 带偏移 GPU buffer 扩容未覆盖完整区间；Python 场景结构缓存键被覆盖，导致每帧重新上传资源。修复容量管理和稳定结构缓存键，保留动态姿态更新。 |
| 轴球转场时文字抖动 | 球心保留小数坐标，文字的 pen 坐标却单独取整。删除取整，让字形顶点和球体连续移动；在 ±X／±Y 点击转场中测量相对位置。 |
| 字体及圆角焦点轮廓锯齿 | 原生 ImGui atlas 误用数据纹理的 point sampler。改为双线性过滤；身份／深度等数据目标仍按数据用途采样。 |
| ANYmal C 报告加载 0.341 秒，但实际等待数秒 | 日志没有包含资源准备和首帧；原生 NumPy mipmap 生成有大量临时数组和串行工作。改用已有 stb resize、释放 GIL 的有界工作池和不可变上传存储，移除重复纹理复制；日志明确分阶段计时。 |
| 点选 MS-Human-700 模型冻结数秒 | Inspector 为收起的全部组件生成字段，重复扫描引用列表近一万次。改为先读声明数量、展开时构建字段，并共享一次查询中的不可变引用候选；长组件列表只绘制可见行。 |
| 关键帧和 MJCF 编辑期间窗口冻结 | UI 中会重建模型的操作进入已有单工作线程队列，UI 保留旧画面和进度，完成回调在 UI 线程执行；错误源码保留原场景。同步 Python Session API 仍保持原语义。 |
| 加载期间 RPC 可能访问正在修改的 Session | 把 RPC pump 放在后台编辑／加载完成检查之后，防止第二个访问入口读取中间状态。 |
| 相机预览阴影与主视口不一致 | peer 没有继承和跟踪主视口阴影质量。创建时及每次预览更新都同步质量。 |
| 原生同步读回及实例提交开销偏高 | 逐 draw 的 Metal staging 上传合并为按布局上传；按请求仅绘制所需数据附件；分割改为一个无损附件；NumPy 保留完成结果的所有权，避免再复制一次。无变化且无独立 overlay 时复用颜色／数据输出，并验证姿态、光照、相机、开关和 resize 失效。 |
| 深度读回仍慢于 OpenGL | 使用 bgfx 对齐 buffer 读回，合并提交／等待为一个 owner job；MSAA 颜色先保留 resolve 结果。去掉 Metal staging texture 到 CPU 的转换，保持 ROI、精度和数组所有权。 |
| 大场景近景仍绘制大量看不见的物体 | 颜色、语义和镜面相机分别保守剔除视锥外实例，阴影继续使用独立光源视锥。姿态、动态 mesh 和负缩放更新包围盒，不删除潜在阴影投射者。 |

地板上的彩色倒影属于模型启用的平面反射，OpenGL 中也存在；本轮没有通过关闭阴影或反射来掩盖错误。所有对照使用相同模型、相机、尺寸、颜色模型及显式功能开关。

## 已覆盖的渲染能力

- 两种颜色模型、ambient、headlight、方向／点／聚光／面积光、image light、雾和 horizon haze。
- 逐实例材质、二维／立方体纹理、线性光 mipmap、负缩放法线、透明排序和可选透明身份输出。
- 方向光级联阴影、局部光阴影、三种质量；平面反射、box 顶面反射和多反射面。
- 天空盒、选择轮廓／xray、3D gizmo、九条 debug 绘制路径、文字、物理诊断可视化。
- 所有现有 RenderFlag、五种 debug view、动态 mesh／skin／flex、tendon 和 actuator 配色。
- 颜色／深度／object ID／分割、RGBA、异步读取、目标寿命、场景独立性、多窗口及资源复用。

Python 公共 API 和行主序契约保持兼容；C++ 仍是渲染和资源基础设施，不把业务扩展搬离 Python。项目自有 C++ 源文件使用 PascalCase，第三方库命名不改动。

## 验收结果与可复现入口

本轮新证据位于 `output/native-refinement/`；前一阶段未受本轮修改影响的专项检查
保留在 `output/native-ui-audit/`，在表中明确标注。性能结果使用 delivery 报告，
中间实验不覆盖、不冒充最终结果。

| 验收 | 结果与证据 |
|---|---|
| `make check` | 1,694 fast、146 integration，分层、静态检查和示例通过；`check-final.log` |
| `make gpu`、`make gpu-wgpu` | OpenGL 443 通过／17 跳过，wgpu 360 通过／8 跳过；`gpu-opengl.log`、`gpu-wgpu.log` |
| 最终 bgfx GPU 回归 | 390 通过／70 跳过；`gpu-delivery-isolated.log`。按仓库要求逐文件隔离；部分文件显式使用 OpenGL。 |
| 私有原生 contracts、render、feature、完整 Viewer | 41 项通过，包含 100 humanoid、ROI 行对齐、并发 ticket、MSAA、数组／目标寿命、VSync 与实际 drawable 数量；`private-final.log` |
| C++ CPU contracts | 5 项 CTest 通过，包含负缩放包围盒和保守视锥判定；`ctest.log` |
| 原生 GPU probe | 最终构建通过；`probe-delivery.log` 和 `probe-delivery/` |
| `make native-parity` | 96 组功能对照通过，含功能开启的实际可见贡献；`parity/` |
| `make native-model-parity` | 16 组刚体、skin、flex、100 humanoid 动态／恢复对照通过；`models/` |
| `make native-motion-parity` | 204 对连续近景平移／环绕／缩放通过；`motion/` |
| `make native-ui-parity` | 最终 318 对窗口图像、100%／150% UI；label 相对误差低于 0.000008 逻辑像素，颜色 P99 差为 0；`ui-delivery/` |
| 响应式 UI 清单 | 112 组中英文／窄宽／缩放布局检查；`layout/`；已查看历史问题对应代表图 |
| `make native-corpus-parity` | 251 个模型 × 8 个视角，18 个空文档，26 个非独立片段；没有未通过的可绘制模型／视角；`corpus/` |
| 真实 Viewer 接入 | `agent-viewer.log` 通过 |
| nanobind／pybind 共存 | 13 项通过，包含 MuJoCo 导入顺序、GIL 和数组寿命；`bindings-final.log` |
| 独立平台 wheel | 最终构建安装后去除开发路径并用 `python -I` 实际渲染、检查依赖许可证；`wheel-delivery.log`、`output/native-wheel/installed.json` |
| Physics、MuJoCo audit、adapter conformance | 前阶段 420 项通过／2 跳过，audit／conformance 通过；`output/native-ui-audit/physics-latest.log` 等，本轮不修改物理实现 |
| Vulkan shader | 前阶段 43 个 shader 编译为 SPIR-V；`output/native-ui-audit/spirv-complete.log`，本轮不修改 shader；不能替代 Vulkan 设备测试 |

最新全集对照位于 `output/native-refinement/corpus/`，已复查 assets、机械臂、四足和人形等代表性总览。
共 2,008 个有物体视角，加上 144 个空场景背景视角，按当前颜色及深度分类身份门槛全部通过。
`ValidationNotes.md` 保留错误单进程 GPU 调用及其原因；最终使用规定的隔离入口，不修改断言门槛。

本机本地语料共发现 295 个模型文档：251 个有可绘制内容、18 个空文档（包括 `assets/empty.xml` 和只含定义的片段）、26 个不能独立编译的 include 片段。18 个空文档通过加载、组合和背景检查，但不计入机器人渲染数量；26 个片段单列为 skipped，不计作通过。这修正了先前把 269 个可加载文档统称为完整模型的口径。

模型语料默认单采样，单独检查物体内部颜色，保留整幅 RGB 差异和原始身份差异；MSAA 边缘由 motion、feature 和真实 Viewer 检查覆盖。11 个视角的原始分割差异超过 0.1%，来自重叠／等深手部几何和远裁剪边界：两个后端的投影深度差在四个 float32 深度缓冲 ULP 内。报告保留 `segmentation_disagreement`，分别记录 `depth_tied_identity`、`far_clip_identity` 和未解释差异，不将其描述为逐像素身份完全一致。其他视角不需要这一分类才能通过原始分割门槛。

连续相机对照的最大整幅颜色平均差异为 0.0213／255，P99 为 0，最大逐帧误差变化为 0.0404／255。已经查看近景方块、球面、阴影／反射、模型总览和真实 Viewer 的输出图。动态恢复测试同时检查相机／姿态恢复后的输出，防止缓存留下旧画面。

代表性文件：

- `output/native-refinement/motion/mujoco-classic-msaa4/pan-08-comparison.png`：左 OpenGL、右 bgfx，检查方块侧面。
- `output/native-refinement/motion/mujoco-classic-msaa4/motion.webp`：连续运动对照。
- `output/native-viewer/projection-bgfx-820-880.png`：窄窗口的实际投影。
- `output/native-refinement/models/100_humanoids-shaded-1-comparison.png`：三后端大量刚体对照。
- `output/native-refinement/corpus/contact-00.png` 至 `contact-13.png`：模型总览，每对左 OpenGL、右 bgfx。

## 历史 UI 与交互问题逐项清点

本次重新运行三后端 GPU 输入回归和 112 组响应式布局检查。截图位于
`output/native-refinement/layout/`，包含中英文、100%／150% UI 缩放和窄面板。
下表把历史反馈对应到代码行为与验证，不将旧截图或单纯编译成功当作验收。

| 历史反馈 | 当前行为与验证 |
|---|---|
| ImGui 圆角、字体和设计规范 | 主题圆角为 4.8；JetBrains Mono 与 Noto Sans CJK。间距、行高、搜索框、badge 和折叠标题使用共用控件，未为这些布局修改第三方 ImGui 核心。 |
| Viewport 三边黑线、浮动窗口 resize 边缘被覆盖 | 停靠内容按视口内容矩形铺满；浮动窗口保留装饰与 resize 区域。GPU 输入测试验证 splitter 不触发场景操作。 |
| Status 的 Ctrl／Drag／Steps 不齐、Δt 左侧空白 | 统一字形基线、badge 内边距和实际文本宽度；窗口变窄时逐级折叠，保留相同分隔间距。`status.png` 与基线／窄栏 CPU 回归。 |
| Pan 鼠标图标空白、鼠标操作写死 | 鼠标图标展示对应按键；状态提示读取统一 pointer action。时间线右拖平移、滚轮缩放有 GPU 行为回归。 |
| Output 文字贴背景、Copy all／Clear 拥挤 | 日志行有内边距，搜索图标位于独立区域，按钮随窄窗口换行。已检查 `output-180-layout.png`、搜索输入与清除测试。 |
| Hierarchy 贴左、展开三角视觉偏下 | 共用行内边距、抗锯齿 disclosure、光学中心；名称／类型／可见性列不互相覆盖。选中与 hover 背景保持一致。 |
| 场景点选后 Hierarchy 留下第二个选中项 | Session 的外部单选同步清除旧批量选择；GPU 回归覆盖虚拟列表与选择状态。 |
| 双击 focus 后取消选择 | 聚焦只改变相机；`test_joint_focus.py` 覆盖 link／joint 聚焦保留选择及双击路径。 |
| Control／Joints 名称和背景局促 | 行高与标签 inset 统一，长名称裁剪并保留 tooltip；搜索、长列表虚拟化和选择操作由三后端回归覆盖。 |
| Settings 左栏尺寸不一、菜单太窄、File 左空白 | 设置导航行统一；窄 Settings 切换成类别下拉框；菜单保留行高和左右 inset。已检查 `settings-360-general.png`、`menu-bar.png`、`window-menu.png`。 |
| Inspector badge 偏心、Transform 独有背景、窄宽截断 | 共用轴输入和 section 布局；先将标签独占一行，再将 X／Y／Z 堆叠，内容高度跟随实际行数。已检查 230 宽、150% 中文的 position／rotation 完整显示。 |
| Type value 前的 Δ 不齐 | 精确输入使用明确的 Relative／Absolute 模式控件，移除孤立 Δ。已检查 `precise-input.png`。 |
| Keyframes 字段随意摆放 | 模型字段、录制／快照按钮、状态文本按剩余宽度布局；230 宽下工具条换行、轨道名称换行，时间线仍可操作。 |
| Assets 展开风格不同、缺少 table | 与 Inspector 共用折叠标题与字段布局，资产表使用 Name／Type／Used 列。已检查 `search-assets.png`，三后端覆盖资产编辑行为。 |
| View cube 文字转动抖动、圆角与字体无抗锯齿 | 标签保留小数坐标，原生 ImGui 字体 atlas 双线性采样；318 对真实窗口图像检查转场中的相对位置和边缘。 |

已实际查看以上代表性截图，以及多模型人体／Go2、近景方块、G1 近景和模型总览。
布局回归通过不等于所有屏幕配置都已人工遍历；120 Hz 和其他操作系统的设备边界见后文。

## 刷新率和交互延迟

没有固定 60 FPS 上限。开启 VSync 时由 GPU／系统当前显示模式安排呈现，不按显示器宣传的最高频率在 CPU 上 sleep。手动平移、环绕和缩放取消聚焦动画，当前相机在同一帧提交给 renderer。

`RenderRuntime` 保留专用原生 owner，bgfx 在该 owner 上处理命令，删除第二层 CPU 帧队列。GPU 最多两帧在途以保留 CPU／GPU 重叠。Metal 开启 VSync 时用两个 drawable，关闭时用第三张备用图像，避免 GPU 已完成但合成器仍占用图像时完全串行等待。多窗口共享同步策略；有任一同步窗口时各 surface 都保持同步和两个 drawable。实际 Metal 属性切换、窗口 resize 和 peer 关闭由原生 Viewer 回归验证。

本轮显示环境为 Apple M5、1× framebuffer，系统及 AppKit 当前报告 **60 Hz**。没有改变系统显示设置。窗口 1200×800 设备像素，场景固定 1306×1036；每项平移／环绕／缩放两轮、每轮三秒，全程前台、未遮挡。表中 FPS 是六轮 CPU 提交 FPS 的均值，P95 是六轮 P95 的中位数。

| 关闭 VSync 的可见窗口 | 提交 FPS | 单帧 P95 | 最大单帧 |
|---|---:|---:|---:|
| 原生，一帧 GPU／两张 drawable 对照 | 261.85 | 8.383 ms | 17.132 ms |
| 原生，最终两帧 GPU／三张 drawable | 371.65 | 7.239 ms | 15.500 ms |
| OpenGL，同尺寸前台对照 | 371.36 | 4.885 ms | 9.205 ms |

最终原生吞吐比一帧限制提高约 42%，与本轮 OpenGL 接近，但 P95 仍较高，不能据此宣称全部交互延迟相同。另一轮相同配置测得约 404 FPS；报告采用最终条件切换复测的 372 FPS，保留轮间波动。`output/native-refinement/window-delivery-bgfx/`、`window-focused-opengl/` 和 `PresentationExperiments.md` 包含证据和全部中间配置。

开启 VSync 的原生约 60.18 次提交／秒，与当前显示模式一致。OpenGL 约 120.72 次 CPU 提交不能证明物理显示有 120 FPS。这些指标不是输入到光子的硬件测量。前一轮 2× framebuffer 的数据、当前较小的 653×518 viewport 和失焦轮次分开保留，不混入上表。真实 120 Hz／HiDPI 设备呈现与尾延迟仍是设备验收项。

## 模型加载速度与计时口径

`make native-load-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` 在已初始化的 Viewer 中通过实际异步队列加载模型。每个模型／后端使用独立进程，首轮与同进程两次重载分列；没有清除系统文件缓存。计时从请求加载到已提交窗口图像可由 GPU 读回，不额外 capture 重绘，也不是物理屏幕扫描时间。本轮 viewport 为 522×400，原始数据与首帧图在 `output/native-refinement/loads-delivery/`。

| 模型 | OpenGL 首次 | bgfx 首次 | OpenGL 重载中位数 | bgfx 重载中位数 |
|---|---:|---:|---:|---:|
| anybotics_anymal_c | 414.6 ms | 344.4 ms | 198.0 ms | 173.7 ms |
| anybotics_anymal_b | 532.2 ms | 473.3 ms | 51.5 ms | 46.3 ms |
| boston_dynamics_spot | 434.1 ms | 374.3 ms | 28.4 ms | 32.4 ms |
| unitree_go2 | 270.3 ms | 233.8 ms | 27.8 ms | 37.6 ms |
| unitree_g1 | 145.8 ms | 124.9 ms | 36.0 ms | 44.6 ms |
| unitree_h1 | 102.3 ms | 92.8 ms | 31.6 ms | 31.3 ms |
| franka_emika_panda | 362.1 ms | 312.0 ms | 38.9 ms | 39.3 ms |
| franka_fr3 | 319.5 ms | 275.5 ms | 31.5 ms | 35.4 ms |
| universal_robots_ur5e | 328.5 ms | 275.8 ms | 28.0 ms | 37.9 ms |
| google_robot | 115.4 ms | 98.4 ms | 35.2 ms | 41.3 ms |

旧原生 ANYmal C 的独立诊断为 3.571 秒，其中 NumPy mip 处理约 2.866 秒；最终首次为 344.4 ms、重载中位数 173.7 ms，OpenGL 为 414.6／198.0 ms。首次十个模型全部更快，重载仍并非全部更快，例如 G1 为 44.6 对 36.0 ms。计时没有把已有渲染器初始化排除后的结果称为整个进程启动速度。

日志包含工作开始到首帧提交的总时间，分列 source／resources／first frame submitted；GPU 就绪由 benchmark 另外记录。资源切换仍有集中上传阶段，不能把后台编译等同为零 UI 停顿。

## MS-Human-700、多模型和编辑操作

`make native-editor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` 重现：新建 workspace → 添加地面 → 添加 MS-Human-700 → 选择模型。继续覆盖模型放置预览／取消、关键帧创建／读取／删除、添加 Go2、切换模型、修改 MJCF、移除及 Undo／Redo。选择操作断言 renderer `set_scene()` 为零次，防止只改变选择却重传资源。

旧实现首次点选 2,235 ms，多模型切回 2,406 ms，源码编辑后再选 2,528 ms。最终可见窗口结果在 `output/native-refinement/editor-delivery-bgfx/` 和 `editor-delivery-opengl/`：

| 操作，到更新窗口提交 | bgfx | OpenGL |
|---|---:|---:|
| 首次点选 MS-Human-700 | 9.46 ms | 9.26 ms |
| 重复点选，中位数 | 5.85 ms | 6.14 ms |
| 多模型切回人体 | 7.13 ms | 8.66 ms |
| 源码编辑后再选 | 10.83 ms | 8.64 ms |
| 平移人体，60 帧中位数 | 4.89 ms | 5.73 ms |
| 同步公共 API 创建快照 | 275.92 ms | 201.31 ms |
| 同步公共 API 删除快照 | 242.24 ms | 178.73 ms |
| 同步公共 API 应用源码 | 948.80 ms | 841.21 ms |

UI 的后台创建／删除快照／应用源码分别为 331／349／1002 ms，过程中更新了 54／64／361 帧；最长 UI 帧约 41／54／33 ms。连续数秒冻结已修复，但这些长帧仍需保留在性能边界中。同步 Python API 保持完成后才返回，不把“已排队”冒充“已完成”。

已检查人体／Go2 的组合图和组件表。Inspector 先读取声明数量，展开后才生成字段，700 项组件只提交可见行；引用候选在一次查询内复用，模型替换后重新读取，避免旧名字或跨模型引用。

## 公共 Renderer 同步输出

`output/native-refinement/renderer-delivery.json` 使用公共 `update_scene()` 和 `render()`，包含 7 类工作负载、640×480／1280×720／1920×1080、RGB／深度／分割、4×颜色 MSAA，共 126 个隔离进程测例。每例预热 10 帧、采样 80 帧；RGB 和深度复用 `out`，分割双方都返回新数组。统计更新加 GPU 结果可读的总时间。

| 输出 | bgfx／OpenGL 耗时比中位数 | 最低～最高 | bgfx 更慢的测例 |
|---|---:|---:|---:|
| RGB | 0.523 | 0.411～0.785 | 0／21 |
| 深度 | 0.543 | 0.387～0.915 | 0／21 |
| 分割 | 0.481 | 0.279～0.880 | 0／21 |

比值小于 1 表示原生更快。静态场景可复用结果；动态场景真实更新。深度保持 R32F，分割保留两个 int32 的语义值，没有降低精度或删除效果。GPU 同步读回使用 bgfx 对齐 buffer；行 padding 在输出边界移除，返回数组独立持有结果，复用／resize／关闭不使它失效。异步请求仍先捕获提交的帧，再等待结果，不被后续绘制替换。

早期直接从 MSAA 颜色资源复制到 buffer 的实验产生黑图，已被 GPU 回归捕获并撤回；最终版本先复制 resolve 图像再读 buffer。`ReadbackExperiments.md` 明确标记了无效实验，以上只使用修正后的最终构建。63 组优势只覆盖这份矩阵，不能外推为所有编辑、所有硬件都更快。

## 物理并行和大量刚体

最终生产 Session 的 100 humanoid（1,600 移动 body、2,700 自由度）使用相同
696×518 viewport、隐藏窗口、VSync 关闭、目标 120 FPS；每轮三秒，每种模式三轮。
原始结果在 `output/native-refinement/physics-delivery-opengl/` 和 `physics-delivery-bgfx/`。

| 后端／物理模式，三轮中位数 | 渲染提交 FPS | 单帧工作 P95 | 物理 step/s |
|---|---:|---:|---:|
| OpenGL，串行 | 73.49 | 17.28 ms | 198.27 |
| OpenGL，并行 | 119.89 | 7.42 ms | 199.49 |
| bgfx，串行 | 105.62 | 10.64 ms | 198.62 |
| bgfx，并行 | 119.78 | 5.59 ms | 199.30 |

每轮仿真状态都与从相同初始状态逐步串行重放完全一致。并行原生 P95 比 OpenGL
低约 25%，双方都接近测试调度上限；不能把 120 调度上限外推成无上限渲染吞吐。
隐藏窗口结果用于隔离 Session／物理工作，不代表屏幕呈现延迟。

本轮此前还复测了轻量刚体和 cloth stress，见 `physics-opengl/`、`physics-bgfx/`。
cloth 并行 P95 为 6.35／4.38 ms，串行为 34.70／26.77 ms，轨迹校验通过。
该组三工作负载在最后呈现队列调整前完成；上表 100 humanoid 使用最终构建。
前一阶段 2× framebuffer 的数字保留在 `output/native-ui-audit/`，不与本轮直接横比。

## 平台与交付边界

bgfx 直接使用 Metal／D3D12／Vulkan，不经过 wgpu 或 Dawn。Linux 构建路径选择 Vulkan，当前窗口路径为 X11／XWayland；尚未提供原生 Wayland 窗口接入。Windows D3D12、Linux Vulkan 和 120 Hz 显示模式仍需要对应环境实测；本轮 SPIR-V 编译不能替代设备验收。wheel 匹配本机 CPython ABI、arm64 和 macOS deployment target，不是跨平台通用包。

## 物理能力、协议和鼠标映射

模型菜单、文件对话框、拖入和加载队列按 `AdapterCaps.model_formats` 提供入口，
没有声明 MJCF / URDF 支持就不提供。扰动、控制写入、拓扑编辑、MJCF 源码、组件及
模型关键帧分别检查能力。Session 在物理同步栅栏和历史捕获之前拒绝不支持的命令；
失败的事务仍整体回滚。仅有 simulation 不代表可以写 ctrl 或控制外部时钟。

扩展通过名称和协议修订号识别，backend_version 仅用于诊断。JSON RPC 的操作发现
包含 version 和 availability，版本不一致在分派前失败。snapshot stream 独立协商
protocol_version / command_versions，取发布端能力和接收端已实现命令的交集；旧发布端
没有声明写回协议时保持只读。远程文件导入和源码编辑不通过这个快照通道暴露。
其他物理适配器需要按自身实际版本声明能力，不能只靠引擎名称推断。

30 项语义鼠标操作与键盘映射保存在同一份配置：覆盖视口、扰动、时间轴、层级、
数值控件、属性复制、Output 和工具栏。支持修饰键、左右键组合、双击和滚轮。
Blender / Unity / Unreal / MuJoCo 预设只调整导航组合，未声称复制完整编辑器或
MuJoCo 的水平／垂直拖动算法。默认行为继续兼容；冲突配置原子拒绝。
普通 ImGui 控件激活和平台文本编辑保留控件约定。


## G1 独立 world 渲染压力

数据固定为 Unitree 官方 Apache-2.0 的 [dance1_subject2.csv](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/motions/g1/dance1_subject2.csv)，提交
`1425b15f73bd4095f0df53709d7c389c3eb9e790`，SHA-256
`1793edcd8345fa4736676c06008186d1a3ecaaf99df0c380218eaa4e0de100ec`。
这是 3,945 帧／30 Hz 的动作，经根旋转插值和关节运动学生成 120 Hz 姿态。
每个 world 使用固定随机种子的独立动作相位，方阵间距 7 m，仅作显示偏移，
没有跨 world 碰撞，也没有声称执行 4,096 份物理仿真或强化学习。

先通过 1／4／16 个机器人、三个近景姿态的 RGB、深度、object ID 和分割对照，
再计时 1024／2048／4096 个机器人。每档原始／简化网格都完成 9 组对照，
图像已检查。GPU 测试和性能采样串行；1280×720、4× MSAA、预热 5 帧、采样 30 帧，
每种后端独立进程。以下 FPS 包含姿态更新、renderer 更新及完整 GPU RGB 读回。

最终原始精度结果在 `output/native-refinement/g1-delivery-overview/` 和
`g1-delivery-detail/`，两组分别先通过自己的 9 组图像 parity。

| world 数 | 全景 OpenGL | 全景 bgfx | 近景 OpenGL | 近景 bgfx |
|---:|---:|---:|---:|---:|
| 1024 | 1.880 FPS | 2.120 FPS | 1.834 FPS | 5.639 FPS |
| 2048 | 0.944 FPS | 1.057 FPS | 0.914 FPS | 2.904 FPS |
| 4096 | 0.472 FPS | 0.529 FPS | 0.448 FPS | 1.489 FPS |

全景把整支方阵放入画面，近景只观察方阵中央，但仍保留全部 world、原始网格和潜在阴影。
颜色、语义和反射 pass 分别使用自己的相机剔除；4096 份近景排除 284,165 个跨 pass
的实例候选，全景排除为零。近景原生约为 OpenGL 的 3.3 倍，全景约快 12%；
这两种不同构图的 FPS 不能互相冒充，原始精度仍不具备实时交互速度。

此前的显式 LOD 验证保留如下。这些是同机前一阶段的数据，不能冒充本轮最终原始质量：

| 显式网格质量 | world 数 | OpenGL FPS | bgfx FPS |
|---|---:|---:|---:|
| LOD .01 / .05 | 1024 | 27.148 | 29.433 |
| LOD .01 / .05 | 2048 | 13.932 | 15.122 |
| LOD .01 / .05 | 4096 | 7.076 | 7.592 |
| 激进远景 LOD .001 / .2 | 1024 | 65.855 | 68.793 |
| 激进远景 LOD .001 / .2 | 2048 | 34.978 | 36.578 |
| 激进远景 LOD .001 / .2 | 4096 | 18.340 | 18.836 |

原始机器人每份 393,270 个三角形；4096 份共有约 16.1 亿个三角形／场景 pass。
35 份 mesh 数据共享，原始唯一网格约 10.5 MiB；
最后一帧全部原生 pass 合计 181 次 draw，不能与 OpenGL 只统计颜色 bucket 的数字直接比较。
实例化已经避免逐机器人调用 draw，但并没有减少 GPU 对三角形的处理。

阴影保持光源视锥独立剔除；全景 4096 份最后一帧保留 283,363 个阴影实例，
排除 146,720 个完全在各级光源视锥外的实例。不能用主相机可见性删除潜在阴影投射者。
移动和动态网格更新刷新包围盒，负缩放／剪切采用保守包围盒。原生 4096 份全景峰值
RSS 约 1,011 MiB，近景约 900 MiB，网格仍共享。

显式 LOD 使用仓库已锁定的 meshoptimizer，法线和 UV 参与误差约束。两档实际约
20,967 和 6,327 个三角形／机器人。**激进档近景出现明显的头部轮廓、关节和表面细节损失**；
中等档也不等同原始质量。它们只作为主动选择的测试，不覆盖原始资产，不自动开启，
也没有实现按屏幕误差自动切换 LOD。各档结果不能当成等质量加速对比。
对应原始／中等／激进图分别在 `g1-worlds-final/`、`g1-worlds-lod-final/`、
`g1-worlds-coarse-final/` 的 `parity-1-bgfx/frame-0.375.png`。

## 独立进程远程监控压力

以下为前一阶段已经完成的协议／监控验收，本轮未修改传输实现，也未重测这些数值。发布进程目标 120 Hz，接收目标 30 Hz，传输为 localhost TCP。
纯传输测试 1024／2048／4096 个 world 的接收频率均约 29.94 Hz，快照年龄 P95 分别
10.80／10.29／8.62 ms。4096 个 world 的每次姿态约 6.56 MiB；这不是 WAN 或训练吞吐。
接收端使用最新快照，覆盖过时帧，不积累播放队列。纯传输证据在 `output/g1-worlds/transport-*.json`。

完整链路另外包含 renderer 更新、绘制和 RGB 读回，8 秒采样，使用同一激进远景 LOD：

| world 数 | OpenGL 完成 FPS | bgfx 完成 FPS | OpenGL 图像年龄 P95 | bgfx 图像年龄 P95 |
|---:|---:|---:|---:|---:|
| 1024 | 30.00 | 29.99 | 59.90 ms | 35.13 ms |
| 2048 | 29.98 | 29.99 | 66.75 ms | 33.87 ms |
| 4096 | 17.11 | 17.41 | 107.17 ms | 72.19 ms |

图像年龄从发布姿态的本机单调时钟时间到 GPU 结果可读取，不是输入到屏幕的光子延迟。
原始精度 4096 份也独立测量：bgfx 开始绘制时快照年龄 P95 为 13.26 ms，图像完成
年龄 P95 为 1.860 秒、约 0.544 FPS；OpenGL 对应约 2.044 秒、4.113 秒、0.484 FPS。
原始精度仅分别完成 5／4 帧，P95 样本很少，只能作饱和瓶颈诊断。
这说明传输快不代表画面完成快，4096 份原始几何仍明显不能交互。
完整监控证据在 `output/g1-worlds-monitor/` 与 `output/g1-worlds-monitor-original/`。
