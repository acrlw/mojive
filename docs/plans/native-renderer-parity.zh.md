# 原生渲染完整性与 parity 验收

本轮在 `codex/cpp-foundation` 修复并验证 Python Viewer 使用的 C++ / bgfx 后端，已同步 `origin/main` 的 `b4887d7`（本地合并提交 `c2260c8`），包含跟踪坐标轴、时间线和 take 视频等更新。可以用 `make native-viewer` 启动。加载、编辑器交互和大部分图像输出已改善，但可见窗口无同步呈现仍有长帧，部分同步深度读回仍慢于 OpenGL，尚未达到所有路径都不慢于 OpenGL 的要求。Linux、Windows 和 120 Hz 实际呈现尚未完成设备验收。本轮没有启动 workflow CI，也没有合并到 main。

## 本轮修复的具体问题

| 用户可见问题 | 原因和修复 |
|---|---|
| 近景方块侧面条纹、平移闪烁，球面出现斜切暗区 | 阴影 shader 手动读取矩阵下标，经 HLSL／Metal 编译后取错深度轴，导致背光判断和 slope bias 错误。改为基向量乘矩阵，统一数学含义，并检查编译后的 Metal shader。 |
| 实际 Viewer 与离屏对照的构图、形状不同 | Viewer 的相机默认 aspect 为 1，原生 backend 没有按目标尺寸修正。现在由目标统一计算投影，resize 同步更新，并新增真实 Viewer 宽／窄／恢复对照。 |
| 空背景颜色不一致 | C++ 默认背景及 8-bit clear 转换与 OpenGL 不同；统一背景和舍入。材质、光照和色彩空间继续通过分对象对照检查。 |
| 开了 VSync 仍可能撕裂，拖动排队感明显 | 原实现只在 CPU sleep，未启用 GPU 呈现同步。现在 VSync 直接传递到 GPU；删除软件刷新率缓存和 sleep，把 bgfx 提交队列限制为一帧、呈现表面保留两个 drawable。 |
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

诊断保存在 `output/native-ui-audit/`。下面列出当前功能验收和对应证据；性能表另外标注采集阶段。

| 验收 | 结果与证据目录 |
|---|---|
| `make check` | 1,694 fast、146 integration，分层、静态检查和示例通过；`check-final.log` |
| `make gpu`、`make gpu-wgpu` | OpenGL 443 通过／17 跳过，wgpu 360 通过／8 跳过；`gpu-opengl-latest.log`、`gpu-wgpu-latest.log` |
| bgfx GPU 回归 | 390 通过／70 跳过；`gpu-native-final.log`。部分文件显式使用 OpenGL，跳过项包括专属其他后端的行为。 |
| 输入映射与 UI | 三后端各 118 项通过；`pointer-final-native.log`、`pointer-final-opengl.log`、`pointer-final-wgpu.log`。 |
| 控制与真实 Viewer 接入 | `agent-control-final.log`、`agent-viewer-final.log` 通过。 |
| main 新功能 | 跟踪坐标轴、录制及 UI 交互 33 项通过；`main-features.log` |
| Physics、MuJoCo audit、adapter conformance | 420 项通过、2 项跳过，audit／conformance 通过；`physics-latest.log`、`mujoco-audit-latest.log`、`conformance-latest.log` |
| C++ CPU contracts | 5 项 CTest 通过，包含 mipmap、负缩放包围盒和保守视锥判定；`culling-build.log` |
| 私有原生 contracts、render、feature、完整 Viewer | 39 项通过，包括 100 humanoid、关闭时未完成读回、回调关闭、移动阴影投射者和网格简化；`private-final.log` |
| `make native-parity` | 96 组功能对照通过，含功能开启必须有实际可见贡献；`parity-culling/` |
| `make native-model-parity` | 16 组刚体、skin、flex、100 humanoid 动态／恢复对照通过；`model-parity-culling/` |
| `make native-motion-parity` | 204 对连续近景平移／环绕／缩放通过；`motion-culling/` |
| `make native-ui-parity` | 100%／150% UI 的 318 对窗口图像；label 最大相对误差低于 0.000008 逻辑像素，颜色 P99 差为 0；`ui-complete/` |
| `make native-corpus-parity` | 251 个模型 × 8 个视角，18 个空文档，26 个非独立片段；没有未通过的模型／视角；`corpus/` |
| Vulkan shader | 43 个 shader 编译为 SPIR-V；`spirv-complete.log` |
| 独立平台 wheel | 安装后移除开发环境变量，实际渲染并检查依赖许可证；`wheel-final.log` |
| nanobind／pybind 共存 | 13 项通过，包含 MuJoCo 导入顺序、GIL 和数组寿命；`binding-complete.log` |

最新全集对照位于 `output/native-ui-audit/corpus/`，已检查全部 14 张总览图。共 2,008 个有物体视角，加上 144 个空场景背景视角，按当前颜色及深度分类身份门槛全部通过。

本机本地语料共发现 295 个模型文档：251 个有可绘制内容、18 个空文档（包括 `assets/empty.xml` 和只含定义的片段）、26 个不能独立编译的 include 片段。18 个空文档通过加载、组合和背景检查，但不计入机器人渲染数量；26 个片段单列为 skipped，不计作通过。这修正了先前把 269 个可加载文档统称为完整模型的口径。

模型语料默认单采样，单独检查物体内部颜色，保留整幅 RGB 差异和原始身份差异；MSAA 边缘由 motion、feature 和真实 Viewer 检查覆盖。11 个视角的原始分割差异超过 0.1%，来自重叠／等深手部几何和远裁剪边界：两个后端的投影深度差在四个 float32 深度缓冲 ULP 内。报告保留 `segmentation_disagreement`，分别记录 `depth_tied_identity`、`far_clip_identity` 和未解释差异，不将其描述为逐像素身份完全一致。其他视角不需要这一分类才能通过原始分割门槛。

连续相机对照的最大整幅颜色平均差异为 0.0213／255，P99 为 0，最大逐帧误差变化为 0.0404／255。已经查看近景方块、球面、阴影／反射、模型总览和真实 Viewer 的输出图。动态恢复测试同时检查相机／姿态恢复后的输出，防止缓存留下旧画面。

代表性文件：

- `output/native-ui-audit/motion-complete/mujoco-classic-msaa4/pan-08-comparison.png`：左 OpenGL、右 bgfx，检查方块侧面。
- `output/native-ui-audit/motion-complete/mujoco-classic-msaa4/motion.webp`：连续运动对照。
- `output/native-viewer/projection-bgfx-820-880.png`：窄窗口的实际投影。
- `output/native-ui-audit/model-parity-complete/100_humanoids-shaded-1-comparison.png`：三后端大量刚体对照。
- `output/native-ui-audit/corpus/contact-00.png` 至 `contact-13.png`：模型总览，每对左 OpenGL、右 bgfx。

## 刷新率和交互延迟

没有固定 60 FPS 上限，也没有根据显示器“最高规格”强行提交的定时器。开启 VSync 时由 GPU／系统的当前显示模式安排呈现；切换显示模式后不依赖缓存的刷新率。实际检查 CAMetalLayer 的 `displaySyncEnabled` 为开／关／开，并验证关闭同进程的另一个窗口不会丢失剩余窗口的同步策略。

本次系统报告 Apple M5、AG25Q380，4096×2304 像素、2048×1152 逻辑分辨率，当前为 **60 Hz**，AppKit 对唯一屏幕报告的 `maximumFramesPerSecond` 也为 60。因此不能宣称已验证 120 Hz 实际呈现。没有修改用户的系统显示设置。OpenGL 的 CPU 提交计数可能高于显示刷新率，不能据此认定显示器运行在 120 Hz。

`make native-window-benchmark` 在可见窗口中连续改变相机，验证变更在同一帧交给 renderer，同时记录每帧窗口遮挡、焦点和实际 Metal VSync 状态。这是 CPU 提交节奏，不是输入到光子的硬件延迟测量。手动平移、环绕和缩放会取消聚焦动画，直接更新相机。

最后一次有状态记录的对照位于 `output/native-ui-audit/window-visible-repeat/report.json`。窗口 1200×800 逻辑像素、viewport 1306×1036 设备像素，每轮三秒，两轮平移均全程聚焦、未遮挡。关闭 VSync：

| 后端 | 两轮提交 FPS | 单帧 P95 |
|---|---:|---:|
| OpenGL | 358.2／355.1 | 5.13／5.02 ms |
| bgfx | 173.3／172.4 | 12.95／12.50 ms |

**这条可见呈现路径没有达到性能要求，不能采用此前约 400 FPS 的短时结果作为稳定结论。** 早期测量没有记录遮挡状态，而 bgfx Metal 在遮挡时会改用离屏目标，不能把这类数据当作真实呈现性能。原始 `window/`、`window-complete/` 和相关异常记录保留，不覆盖或挑选有利轮次。

临时 Metal 分段诊断定位到 `nextDrawable` 等待。增加呈现 drawable 到三张、增加 GPU 工作队列到两帧均没有稳定收益，已恢复一帧 GPU 工作队列和两张呈现 drawable。临时跟踪代码未进入依赖或最终构建。开启 VSync 后原生约 60.3 FPS，与本机当前 60 Hz 显示模式一致；OpenGL 约 120 次 CPU 提交不能证明 120 次物理呈现。仍需继续处理无同步呈现的尾延迟，并在实际 120 Hz 模式测量输入到呈现的延迟。

## 模型加载速度与计时口径

`make native-load-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` 在初始化好的 Viewer 中通过实际异步加载队列加载模型。每个模型／后端使用独立进程，首轮与同进程两次重载分列；操作系统文件缓存没有清除。表中时间从请求加载到**本帧可由 GPU 读回**，不等同于物理屏幕扫描时间。读取的是已提交窗口图像，不额外调用 capture 重新渲染。

| Menagerie 模型 | OpenGL 首次 | bgfx 首次 | OpenGL 重载中位数 | bgfx 重载中位数 |
|---|---:|---:|---:|---:|
| ANYmal C | 794.7 ms | 431.3 ms | 263.8 ms | 310.8 ms |
| ANYmal B | 552.3 ms | 445.7 ms | 60.4 ms | 48.9 ms |
| Spot | 421.6 ms | 387.8 ms | 32.3 ms | 49.1 ms |
| Go2 | 320.7 ms | 258.5 ms | 37.8 ms | 44.1 ms |
| G1 | 204.5 ms | 149.8 ms | 54.5 ms | 46.4 ms |
| H1 | 128.9 ms | 99.1 ms | 38.0 ms | 44.2 ms |
| Panda | 376.7 ms | 355.7 ms | 44.6 ms | 45.8 ms |
| FR3 | 338.4 ms | 296.7 ms | 41.3 ms | 50.4 ms |
| UR5e | 339.0 ms | 304.8 ms | 33.0 ms | 36.3 ms |
| Google Robot | 130.0 ms | 106.3 ms | 38.3 ms | 41.6 ms |

原始数据和实际首帧图在 `output/native-ui-audit/loads/`。旧原生 ANYmal C 的独立诊断为 3.571 秒，其中 NumPy mip 处理约 2.866 秒；不是纯 GPU 同步问题。新实现资源阶段三轮中位数约 94.2 ms，OpenGL 为 209.0 ms。上表首次全部更快，但重复加载并非全部更快，ANYmal C 重载仍有约 47 ms 差距。

日志现在包含从工作开始到首帧提交的总时间，并分列 `source`、`resources`、`first frame submitted`；GPU 就绪等待另由 benchmark 报告。调用提交完成仍不代表用户已经看到屏幕像素，这一边界在文字中明确保留。

## MS-Human-700、多模型和编辑操作

通过 `make native-editor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` 重现：新建 workspace → 添加地面 → 添加 `ms_human_700/MS-Human-700.xml` → 点选模型。还覆盖模型放置预览／取消、快照创建／读取／删除、添加 Go2、切换模型、应用 MJCF、移除模型及 Undo／Redo。每次选择都断言 renderer `set_scene()` 次数为零，防止仅改变选择却重传全部资源。

同机、同窗口尺寸的旧组件实现对照保存在 `editor-before/`。旧实现首次点选 2,235 ms，多模型切回 2,406 ms，源码编辑后再选 2,528 ms。更新后的离屏窗口隔离测量分别约 6.9、12.8、7.5 ms（`editor-bgfx/`）；该组用于隔离编辑器工作，不代表物理呈现延迟。

实际可见窗口的更新结果在 `editor-visible-bgfx/` 和 `editor-visible-opengl/`：

| 操作（到更新窗口提交） | bgfx | OpenGL |
|---|---:|---:|
| 首次点选 MS-Human-700 | 7.35 ms | 12.05 ms |
| 多模型切回人体 | 12.21 ms | 9.68 ms |
| 源码编辑后再选 | 9.22 ms | 8.98 ms |
| 平移人体，60 帧中位数 | 6.50 ms | 6.53 ms |
| 同步公共 API 创建快照 | 274.69 ms | 201.73 ms |
| 同步公共 API 删除快照 | 224.43 ms | 199.96 ms |
| 同步公共 API 应用源码 | 996.36 ms | 829.17 ms |

UI 的后台编辑流程另外记录总时间和最长 UI 帧：原生创建快照 377 ms／59 ms，删除快照 330 ms／42 ms，应用源码 1,005 ms／53 ms；编译期间分别持续更新 34、34、134 帧。后台化减少了连续冻结，但仍有约 40–60 ms 的长帧，不能宣称编辑完全没有卡顿。Python 调用方的同步命令保持兼容，不把“已排队”冒充“已完成”。

已检查 `editor-visible-bgfx/selected-human.png` 和 `composed-models.png`。人体、肌腱、Go2、地面及各模型身份正常；source 查询的引用候选不跨调用缓存，模型替换后能刷新名字；组件 ID、字段和路径的现有编辑／保存／恢复测试保持覆盖。

## 公共 Renderer 同步输出

`output/native-ui-audit/renderer-final.json` 使用公共 `update_scene()` 和 `render()`，包含 7 类工作负载、640×480／1280×720、RGB／深度／分割、4×颜色 MSAA，共 84 个隔离进程测例。每例预热 10 帧、采样 80 帧；RGB 和深度复用 `out`，分割双方都返回新数组。统计更新加同步读回总耗时，性能测试独立运行。

| 输出 | bgfx／OpenGL 耗时比的中位数 | 最低～最高 | bgfx 更慢的测例 |
|---|---:|---:|---:|
| RGB | 0.930 | 0.562～1.143 | 1／14 |
| 深度 | 1.174 | 0.814～1.418 | 11／14 |
| 分割 | 0.760 | 0.554～1.135 | 2／14 |

比值小于 1 表示原生更快。深度的主要剩余差距在 720p 同步读回：例如 dense mesh 为 OpenGL 0.810 ms、bgfx 1.044 ms。动态 1,024 物体的 720p RGB 为 1.787／2.042 ms，深度为 1.129／1.455 ms。**“每条路径都不慢于 Python OpenGL”尚未达到**，不能用编辑器或静态场景的优势代替这个结论。当前没有降低深度精度、删掉渲染效果或更改语义 ID 来换分数。

额外的 720p 单盒诊断把原生路径拆成提交、排队读回、等待、导出 NumPy、复制到 `out` 和释放：深度等待中位数约 0.51 ms，导出数组约 0.001 ms，复制到 `out` 约 0.062 ms。绑定层导出已经不是此例的主要开销，后续优化应聚焦读回调度／GPU 就绪及调用方所需的同步边界。原始分段结果在 `readback-phases.json`；这个简单场景不替代完整矩阵。

静态场景复用渲染结果，因此这里同时包含缓存收益；动态场景每帧真实更新。数据附件按请求裁剪，分割的两个 int32 由一个 RGBA16 UNORM 附件无损保留，深度仍是 R32F；缓存失效和输出数组寿命有实际 GPU 回归测试。

## 物理并行和大量刚体

本轮生产 Session 的三种工作负载使用 1392×1036 viewport、VSync 关闭、目标 120 FPS、每轮三秒、每种模式三轮。结果位于 `output/native-ui-audit/physics-opengl/` 和 `physics-bgfx/`。100 humanoid（1,600 移动 body、2,700 自由度）三轮中位数：

| 后端／物理模式 | 渲染提交 FPS | 单帧工作 P95 | 物理 step/s |
|---|---:|---:|---:|
| OpenGL，串行 | 76.8 | 17.03 ms | 198.4 |
| OpenGL，并行 | 119.7 | 10.03 ms | 199.5 |
| bgfx，串行 | 105.5 | 12.37 ms | 198.4 |
| bgfx，并行 | 119.7 | 5.91 ms | 199.1 |

并行时原生单帧工作 P95 低约 41%，两者均接近 benchmark 的调度上限，不能推算无上限吞吐提升。布料压力场景的并行 P95 为 OpenGL 6.41 ms、bgfx 3.20 ms；轻量刚体为 5.92／4.79 ms。所有仿真均通过逐步串行重放比较。

物理性能采集在本轮最终读回／静态输出缓存优化之前完成；它每帧改变物理状态，不使用静态结果缓存。可见窗口和编辑操作来自后续复测。同步 Renderer 矩阵在阴影视锥剔除后重新测量；旧 `renderer-complete.json` 保留，G1 表另外给出剔除前后结果。旧报告使用不同 viewport，不能与本表直接横比。

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

最近一次真实窗口编辑对照在 `editor-final-bgfx/`、`editor-final-opengl/`。
bgfx 首次点选 MS-Human-700 为 8.36 ms，多模型切回 12.44 ms，编辑后再选 6.84 ms；
OpenGL 分别为 8.52、8.54、8.75 ms。平移 60 帧中位数为 4.89／5.56 ms。
UI 后台创建／删除快照、源码修改仍有最高约 61 ms 的长帧，不能视为零卡顿。
已经检查 `mouse-settings.png` 和 `component-table.png`，700 个组件只提交可见表格行。

## G1 独立 world 渲染压力

数据固定为 Unitree 官方 Apache-2.0 的 `dance1_subject2.csv`，提交
`1425b15f73bd4095f0df53709d7c389c3eb9e790`，SHA-256
`1793edcd8345fa4736676c06008186d1a3ecaaf99df0c380218eaa4e0de100ec`。
这是 3,945 帧／30 Hz 的动作，经根旋转插值和关节运动学生成 120 Hz 姿态。
每个 world 使用固定随机种子的独立动作相位，方阵间距 7 m，仅作显示偏移，
没有跨 world 碰撞，也没有声称执行 4,096 份物理仿真或强化学习。

先通过 1／4／16 个机器人、三个近景姿态的 RGB、深度、object ID 和分割对照，
再计时 1024／2048／4096 个机器人。每档原始／简化网格都完成 9 组对照，
图像已检查。GPU 测试和性能采样串行；1280×720、4× MSAA、预热 5 帧、采样 30 帧，
每种后端独立进程。以下 FPS 包含姿态更新、renderer 更新及完整 GPU RGB 读回。

| 网格质量 | world 数 | OpenGL FPS | bgfx FPS |
|---|---:|---:|---:|
| 原始 | 1024 | 1.936 | 2.177 |
| 原始 | 2048 | 0.972 | 1.089 |
| 原始 | 4096 | 0.487 | 0.546 |
| 显式 LOD .01 / .05 | 1024 | 27.148 | 29.433 |
| 显式 LOD .01 / .05 | 2048 | 13.932 | 15.122 |
| 显式 LOD .01 / .05 | 4096 | 7.076 | 7.592 |
| 激进远景 LOD .001 / .2 | 1024 | 65.855 | 68.793 |
| 激进远景 LOD .001 / .2 | 2048 | 34.978 | 36.578 |
| 激进远景 LOD .001 / .2 | 4096 | 18.340 | 18.836 |

原始机器人每份 393,270 个三角形；4096 份共有约 16.1 亿个三角形／场景 pass。
35 份 mesh 数据共享，原始唯一网格约 10.5 MiB，原生进程峰值 RSS 约 1,014 MiB；
最后一帧全部原生 pass 合计 181 次 draw，不能与 OpenGL 只统计颜色 bucket 的数字直接比较。
实例化已经避免逐机器人调用 draw，但并没有减少 GPU 对三角形的处理。

新增阴影视锥包围盒剔除，在 4096 份场景的最后一帧保留 283,363 个阴影实例、
排除 146,720 个完全在各级光源视锥外的实例（约 34%）。移动和动态网格更新刷新包围盒，
负缩放／剪切使用保守变换；颜色和反射 pass 不变。原始网格 bgfx 从此前
1.854／0.929／0.465 FPS 提升到 2.177／1.089／0.546 FPS，约提升 17%，并超过本轮
同精度 OpenGL。`g1-worlds-final/` 保留剔除前记录，`g1-worlds-culled/` 为更新后结果。

显式 LOD 使用仓库已锁定的 meshoptimizer，法线和 UV 参与误差约束。两档实际约
20,967 和 6,327 个三角形／机器人。**激进档近景出现明显的头部轮廓、关节和表面细节损失**；
中等档也不等同原始质量。它们只作为主动选择的测试，不覆盖原始资产，不自动开启，
也没有实现按屏幕误差自动切换 LOD。各档结果不能当成等质量加速对比。
对应原始／中等／激进图分别在 `g1-worlds-final/`、`g1-worlds-lod-final/`、
`g1-worlds-coarse-final/` 的 `parity-1-bgfx/frame-0.375.png`。

## 独立进程远程监控压力

发布进程目标 120 Hz，接收目标 30 Hz，传输为 localhost TCP。
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
