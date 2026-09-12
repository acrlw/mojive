# 自研 Draw2D：需求、实施清单与 agent 交接

本文是下一位实施 agent 的执行入口，配套[中文架构方案](draw2d-architecture.md)。
状态：待实施清单。不要把文档存在、测试已设计或旧版本的验收通过，当作新功能已经完成。
本文基于 2026-09-12 已提交的代码基线；实施前必须重新核对文件与符号。

## 0. 从这里开始

### 0.1 用户要什么

用户希望重要的 2D 绘图由 Mojive 自己管理，不再依赖 ImGui draw list 的几何处理、抗锯齿和提交。
同时补齐 CLI/RPC 的功能覆盖、批处理与调度缺口，整理模块职责，提高交互流畅性。
保留当前已经反复调好的图标几何、颜色、尺寸和对齐规则，不借渲染迁移重新设计图标。

架构方案和实施记录用中文；代码、注释、日志、提交说明及既有产品 UI 文案遵循仓库英文约定。
不要继续讨论一遍同样的架构后停止；按本文依赖顺序实施，并为每一阶段留下真实验证结果。

### 0.2 已完成与未完成

代码基线为 `9214213de4122df035fd62d2ecb9184ca102caaf`，包含以下已提交内容：

- `7dc70f6`：用户已确认的图标调参及此前图标修复。
- `4451fb1`：Scale 编辑、尺寸预览复用与 MuJoCo 空批次优化。
- `9214213`：有界图标命令缓存与性能统计。

本交接文档位于这些提交之后。接手时使用包含本文件的最新分支；不要退回 `7dc70f6`，
否则会丢掉已经完成的 Scale 和绘图缓存优化。**性能比较以含缓存优化的代码为基线。**

已完成的功能：

- Inspector 的 Scale、`commands.SetScale`、对象与适配器缩放能力、尺寸烘焙。
- Scale 草稿预览、Apply 后恢复 identity、Undo/Redo、混合尺寸编辑的换算。
- 尺寸预览复用缓冲；无实际模型写入的 MuJoCo 批次避免额外编译。
- `ui/icons.py` 的有界绘图命令缓存；颜色与透明度在提交时绑定。
- 相关测试、接口文档和 `ui-runtime --scale-only` 验收入口。

尚未完成：自研 Draw2D、独立路径编译器、合成桥接、新的三后端执行器、Scale RPC、
RPC 查询补齐、请求预算、统一编辑批处理。下一节的任务均不得预先勾选。

### 0.3 接手顺序

1. 读取根目录 `AGENTS.md`、[开发规范](../guides/development.md)、[验证矩阵](../guides/testing.md#change-mapping)。
2. 读取本清单和[架构方案](draw2d-architecture.md)，再按源码导航表阅读当前阶段的文件。
3. 运行 `git status --short`、`git diff --stat`、`git diff --check`，区分已有改动与自己新增的改动。
4. 将实际起点的 commit 和内容校验值保存到 `output/draw2d/baseline/`。
   当前 Scale 测试和绘图缓存均已提交；若接手时又出现工作区改动，还需保存 tracked diff 与相关 untracked 文件副本。
5. 完成 T00 的环境与基线检查，然后从 T01 开始。不要先重写整个 Viewer 或整个 Session。

禁止用 `git reset --hard`、全目录 restore、全仓格式化或全量 stage“清理现场”。
如需要隔离基线，创建独立目录或工作树，并明确带入当前已审核的工作区补丁及新增文件。
临时产物放 `output/`；只关闭自己启动的窗口、服务和子进程。

## 1. 需求列表与完成标准

| ID | 需求 | 判定完成的证据 |
| --- | --- | --- |
| R01 | 独立绘制 | 不导入 ImGui 的进程能构造路径并渲染离屏 RGBA；新路径不能调用 `ImDrawList`、临时 ImGui 剖分或 ImGui AA。 |
| R02 | 三后端一致 | OpenGL、WebGPU、bgfx 对同一绘图数据与交互案例完成验收。默认 OpenGL 不能遗漏。 |
| R03 | 保持图标设计 | 直接使用当前生产图标定义和调参，Icon Library 与生产控件共用路径；无自动重新居中、改 padding、改笔画。 |
| R04 | 正确透明合成 | 同色并集和组透明度分别验证；透明背景上无重叠增厚、接缝或黑边。 |
| R05 | 几何质量 | 孔洞、凹路径、描边、G3、镜像、非均匀变换、亚像素与 DPI 场景通过参考比较。 |
| R06 | UI 顺序 | 裁剪、滚动、浮动窗口、弹窗、模态层和图标/文字顺序保持正确，原生输入和焦点不退化。 |
| R07 | 资源复用 | 同一几何仅改变位置/颜色时不重新编译或上传几何；实例数据更新可以发生。 |
| R08 | 有界生命周期 | 几何/纹理/图层池和请求队列有预算；帧内资源不能提前释放；关闭与失败路径无悬挂任务。 |
| R09 | 易用接口 | 普通 draw 自动形成批次；批量使用 `draw_batch`，同图形多实例使用 `draw_instances`；业务代码不操作 GPU 句柄。 |
| R10 | Scale 控制接口 | CLI/RPC 可以发现、校验并执行 Scale；目标能力、过期文档、错误输入、Undo/Redo 全部覆盖。 |
| R11 | 编辑语义统一 | UI 与 RPC 共用事务/重建策略；失败原子回滚、保留失败原因，不重复缩放，不把预览当成已提交数据。 |
| R12 | RPC 不拖垮交互 | 明确限制排队、消息与连接；每帧同时限制数量和耗时；超时、关闭、过载有确定行为。 |
| R13 | 可测量优化 | 与当前已优化 ImGui 路径进行同条件比较，报告冷/热耗时、交互尾延迟、上传量、GPU 与内存。 |
| R14 | 安装与兼容 | Draw2D 导入不启动图形设备；CPU 编译器可独立构建；缺少原生组件时明确报告能力，不能伪称自研后静默回退。 |

## 2. 命名、模块与接口约束

### 2.1 命名规范

| 范围 | 规范与例子 |
| --- | --- |
| Python | 模块、函数、参数使用 `snake_case`；类型使用 `PascalCase`；常量使用 `UPPER_SNAKE_CASE`；内部实现使用前导 `_`。 |
| C++ | 类型、文件使用 `PascalCase`，例如 `Draw2D.hpp`；函数和字段用 `camelCase`；私有成员用 `m` + `PascalCase`。 |
| 目录 | 小写，多词使用下划线；沿用现有 `render/opengl/passes` 等目录。 |
| 公共绘图 | 使用 `draw`、`draw_batch`、`draw_instances`、`fill_path`、`stroke_path`；同一个概念不要添加多套同义拼写。 |
| 图形术语 | 批处理是 batching；实例化是 instancing；multi-draw 是后端机制；render pass 与 render batch 不混用。 |
| 域接口 | 使用 position、rotation、Scale、body frame/world frame；保留现有 node/object/camera/light ID 区别。 |

不要为了“更专业”引入无意义的 Service/Manager/Registry 层，也不要把所有类型命名为含糊的 data/context/options。
新抽象必须对应实际所有权或两个以上需要共享的具体策略。

### 2.2 建议文件落点

以下“新增”路径尚不存在；先实现最小模块，达到明确职责边界时再拆分，不预建空框架。

| 路径 | 状态与职责 |
| --- | --- |
| `python/src/mojive/draw2d.py` | 新增公共契约与轻量入口；不导入 UI、物理或具体 GPU 后端。 |
| `python/src/mojive/ui/draw2d.py` | 保留旧导入兼容；现有 `ImguiDraw2D` 作为迁移期参考适配器。 |
| `cpp/include/mojive/Draw2D.hpp` | 新增中立原生契约；仅含 Mojive 值类型、资源标识和跨度，不暴露 ImGui/bgfx 类型。 |
| `cpp/src/Draw2D.cpp` | 新增 CPU 路径编译和记录实现；不得依赖图形上下文。 |
| `cpp/bindings/Draw2D.cpp` | 新增批量绑定；遵循现有 nanobind 入口、dtype 校验和 GIL 释放模式。 |
| `python/src/mojive/render/draw2d.py` | 新增渲染侧数据包与提交契约；不能反向导入 UI 适配层。 |
| `python/src/mojive/ui/compositor.py` | 新增 ImGui 导入、插入标记与顺序合并；只处理 UI 桥接，不做路径剖分。 |
| `python/src/mojive/render/opengl/passes/draw2d.py` | 新增 OpenGL 执行器；复用现有设备、目标和资源管理。 |
| `python/src/mojive/render/webgpu/passes/draw2d.py` | 新增 WebGPU 执行器；实现相同数据语义。 |
| `cpp/src/bgfx/` | bgfx 执行器新增私有 Draw2D pass；通过现有 Renderer 接入，不复制窗口/runtime。 |
| `cpp/CMakeLists.txt`、`cpp/cmake/Bindings.cmake` | 纳入 CPU 库、绑定和测试；纯 CPU 构建不能顺带要求 bgfx。 |

### 2.3 必须先写清的接口细节

在 T04 开始时将下表落实成类型、docstring 和契约测试，而不是留给各个后端自行理解：

| 接口 | 输入、输出与约束 |
| --- | --- |
| `Path2D` | 不可变路径；坐标是有限数；操作包括 move/line/quadratic/cubic/arc/close；明确填充规则。路径构建器只在构建阶段可变。 |
| `DrawingBuilder2D.finish()` | 返回不可变 `Drawing2D`，包含局部几何、描边和颜色槽位。修改源数组不应改变已完成图形。 |
| `StrokeStyle` | 明确宽度、cap、join、miter limit 和 `local`/`screen` 宽度空间；无隐含像素偏移。 |
| `Affine2D` | 明确 2D 仿射矩阵布局、乘法顺序和转换规则；建议内部 2×3 行布局。例子中的 `translation @ scale` 表示先缩放再平移。 |
| `frame.draw(...)` | 一个图形实例，附变换、颜色槽位、整体 opacity；整体 opacity 必须遵循组语义，不能下发到每个重叠部件。 |
| `frame.draw_batch(items)` | 有序绘制项，可以不同图形；与依次调用 draw 的图像语义一致。首次实现可接受序列，内部尽快转为连续记录。 |
| `frame.draw_instances(...)` | 同一图形，连续 `[N, 2, 3]` 变换数组、每颜色槽 `[N, 4]` RGBA 数组；长度、类型、有限性一致校验；首版公共数组入口统一使用 float32。 |
| `clip_rect` / 图层作用域 | 使用结构化作用域保存/恢复状态；异常退出恢复状态；finish 时不能存在未关闭图层。空裁剪不产生绘制。 |
| `frame.finish()` | 冻结并返回帧数据；之后不能继续写入该 frame；记录器可以开始下一帧。 |
| 提交 | 数据包带资源租约，CPU 消费和 GPU 完成的生命周期分别定义；异步不能持有无所有权的临时指针。 |
| 错误 | 输入问题使用明确的参数错误；容量、资源失效、后端不可用分别报告。不得忽略损坏索引或静默漏画。 |

保留现有 float RGBA 入口并明确其颜色编码。内部合成使用预乘 alpha，但不能只改变 blend state 而不转换输入。
同一图形多实例不等于只能一次 GPU 调用；透明组、内部多次操作和裁剪都可能要求拆分。
空数组是有效的空绘制；非连续数组可在公共边界统一复制一次，不能在每个顶点处隐式转换。

## 3. 源码与背景阅读导航

路径均相对仓库根目录，优先按符号搜索，避免依赖会变化的行号。

| 要解决的问题 | 从哪里进入 |
| --- | --- |
| 图标参数与布局 | `python/src/mojive/ui/icons.py`：`ICON_TUNING_DEFAULTS`、`ICON_GLYPH_PADDING_DEFAULTS`、`ICON_GLYPH_STROKE_DEFAULTS`、`_production_icon_layout`、`draw_icon`。 |
| 图标调用与胶囊 | `ui/viewport_widgets.py`；`ui/app.py` 的 `_draw_playback_widget`、`_draw_tool_column_widget`、`_draw_context_hint_widget`。 |
| 当前几何和 AA | `ui/draw2d.py`：`_concave_indices`、`indexed_fill`、`_write_anti_alias_fringe`；`curves2d.py` 与 `draglink2d.py`。 |
| UI Feasibility | `design/tools/render_ui_feasibility.py`：Icon Library、生产预览替换和参数导出；不要复制生产 painter。 |
| OpenGL/WebGPU 窗口 | `ui/window.py`、`ui/window_wgpu.py` 的 `end_frame` 与字体/纹理处理。 |
| 原生提交与复制 | `ui/window_native.py` 的 `_draw_packet`、`end_frame`；`cpp/bindings/Render.cpp` 的 `render_ui`；`cpp/src/bgfx/Renderer.cpp` 的 `renderUi`。 |
| 原生契约与线程 | `cpp/include/mojive/Render.hpp`、`RenderRuntime.hpp`、`cpp/src/Contracts.cpp`、`RenderRuntime.cpp`。 |
| 原生 UI shader | `cpp/shaders/vs_ui.sc`、`fs_ui.sc`、`varying.def.sc`；调试 shader 可参考 `fs_debugStroke.sc` 等。 |
| 世界/屏幕调试图形 | `render/debugdraw.py` 的 Layer、PackedFrame；`render/overlay.py`；OpenGL/WebGPU 的 debug pass。 |
| Scale 与草稿 | `commands.py`、`geometry.py`、`model_edits.py`、`model_preview.py`、`session.py` 的 `scale_target`、`scale_factors`、`apply_model_edits`。 |
| 物理与自建对象边界 | `scene.py`、`adapters/base.py`、`adapters/static.py`、`adapters/workspace.py`、`adapters/mujoco_adapter.py`。 |
| CLI/RPC 共用入口 | `operations.py` 的 `Operation`、`_cmd`、`prepare_operation`；`control_schema.py`；`control.py` 的 `dispatch`、`_edit_scene`、`_inspect_object`。 |
| RPC 排队与关闭 | `control_rpc.py` 的 `ViewerControlService`、`_RequestHandler`、`RpcClient`；`ui/app.py` 的 RPC pump 和草稿作用域。 |
| CLI 参数与错误 | `cli.py` 的 `cmd_control`、`cmd_operations`；复用 schema 和结构化错误，不新增第二套解析器。 |

必读背景：[绘图扩展指南](../how-to/ui-drawing.md)、[图标设计与验收](../how-to/ui-icons.md)、
[G3 圆角](../how-to/ui-corners.md)、[RPC 控制](../how-to/rpc-control.md)、
[适配器扩展](../how-to/custom-adapter.md)、[编辑与 MJCF](../guides/editor-and-mjcf.md)、
[原生开发](../guides/development.md)、[原生资源与运行时](native-resource-runtime.zh.md)。

需要操作 Viewer、场景或 RPC 时，先读 `.agents/skills/mojive/SKILL.md` 和
[agent 工作流](../how-to/agent-workflows.md)。已有用户 Viewer 必须连接它自己的 endpoint，
不能启动第二个服务后误认为在操作用户看到的那个窗口。

## 4. 按依赖执行的任务清单

每完成一项，在本节勾选，并补充“代码位置、测试命令与结果、产物路径、剩余限制”。
发现实现已经存在时先验证，不重复实现。下一阶段不代替上一阶段的验收。

### T00：接手、环境与基线（所有任务的前置）

- [ ] 保存工作区基线，核对上一轮 Scale 与图标缓存仍在；运行相关 CPU 检查。
- [ ] 记录 Python/依赖版本、系统、GPU、驱动、UI scale、framebuffer scale、实际 renderer 和窗口可见性。
- [ ] 验证原生 ImGui wheel、CPU 原生扩展与各 GPU 后端可用性；记录缺失项。
- [ ] 跑第 6 节的已有基线入口，保存原始样本和至少一套图标/UI 截图。

完成证据：`output/draw2d/baseline/manifest.json`、现有代码副本或补丁、原始报告与截图。

### T01：Scale 的 CLI/RPC 闭环（R10；依赖 T00）

- [ ] 在 `operations.py` 用现有 `_cmd` 注册 `set_scale`：`node_id`、三个正数 `scale`、可选文档前置条件；要求 paused、`write_scale`，允许事务。
- [ ] 复用 `SetScale` 和 Session 的目标解析，禁止在 control/CLI 中重写缩放乘法。
- [ ] 更新 `control_schema.py` 和 inspection：至少返回 `scalable` 与已提交的 Scale 状态；在文档中说明直接调用会立即烘焙、之后为 identity。
- [ ] 测试发现/schema、父节点与几何节点、无能力目标、非法/非有限数/溢出、过期文档、Undo/Redo、保存重开。
- [ ] 增加 CLI 真正通过 socket 执行的用例，不只测字典里出现了方法名；维护 `docs/how-to/rpc-control.md`。
- [ ] 对照 `commands.py` 与 `operations.py` 输出命令覆盖表，标记公共操作、内部事务和本地交互；补齐本范围内的缺口，记录其他未公开命令的理由。

参考测试：`test_operations.py`、`test_rpc_scene.py`、`test_control_cli.py`、`test_cli_automation.py`、`test_scale.py`。
完成证据：真实创建对象→查询 ID→缩放→查询尺寸→Undo/Redo 的无窗口和 attached-viewer 两条链路。

### T02：统一编辑计划与草稿冲突（R11；依赖 T01）

- [ ] 比较 `ControlApplication._edit_scene` 与 `Session.apply_model_edits`，提取共用执行策略，保留现有返回值格式。
- [ ] 保留真正的事务原子性：先校验可校验项，提交时重新检查文档；失败回滚并保留首个失败原因。
- [ ] 根据操作类型合并后端常量更新/重建；不要把“一个 Undo”误当成“已经只有一次重建”。
- [ ] 保留创建后的真实身份绑定及重建后重新解析，禁止猜 ID 或重用旧索引。
- [ ] 明确 attached RPC 的草稿冲突：首版文档修改遇到未提交 UI 草稿时显式拒绝，不隐式 Apply/Discard；只读查询说明其读取的是已提交状态还是预览。
- [ ] 验证尺寸→Scale→尺寸的混合顺序、单次烘焙、创建后编辑、失败恢复、空适配器、物理适配器。

参考测试：`test_pending_model_edits.py`、`test_scale.py`、`test_operations.py`。
完成证据：同一编辑序列经 UI 和 RPC 得到相同文档，只有一个历史记录；测量后端实际重建/常量更新次数。

### T03：RPC 工作预算与诊断（R12；依赖 T00，可在 T02 前独立完成）

- [ ] 给队列、消息长度和连接数增加明确且可配置的预算；结合已有 MJCF/编辑消息测量制定默认值，避免任意小限制破坏合法操作。
- [ ] pump 同时限制请求数和经过时间；预算在请求之间检查，不宣称可抢占已运行的 handler。
- [ ] 队列满返回 `busy`；超大消息在完整 JSON 解析之前拒绝；不能截断后继续当成另一条消息解析。
- [ ] 保留截止时间、取消、`completion_unknown`、关闭排空和“不自动重试修改”的语义。
- [ ] 对昂贵准备任务复用已有 worker，提交在所属线程完成；禁止把 Session 任意丢给线程池并发修改。
- [ ] 添加有界 RPC 统计与读取入口，继续使用 `operations.py`；测试突发请求、断线、截止前/执行中超时、关闭和大消息。

参考测试：`test_rpc_lifecycle.py`、`test_rpc_client_validation.py`、`test_control_rpc.py`。
完成证据：已拒绝、排队超时或取消的未启动请求不得写入场景；耗尽本帧预算但仍有效的请求留待后续帧。
队列可排空；Viewer 在请求压力下仍能处理输入，记录尾延迟。

### T04：中立契约、CPU 构建与记录器（R01/R09/R14；依赖 T00）

- [ ] 按第 2 节实现公共类型、路径构建器、帧记录器及旧导入兼容，不引入第二套 Viewer/Device。
- [ ] 在 CPU-only 构建中加入原生编译器及绑定；验证导入没有加载 ImGui、GLFW、MuJoCo 或 GPU 设备。
- [ ] 定义并测试矩阵布局、数组 dtype/所有权、paint slot、填充规则、状态恢复、finish 后行为和索引校验。
- [ ] 为记录器生成可检查的有序命令转储，用于测试顺序及后续性能诊断。

建议新增测试：`python/tests/test_draw2d_contracts.py`、`cpp/tests/Draw2D.cpp`、`python/binding_tests/test_native_draw2d.py`。
完成证据：CPU-only 进程构造并冻结路径/帧，输入变更与释放不改变已发布数据。

### T05：路径、描边和独立剖分（R01/R05；依赖 T04）

- [ ] 复用现有 G3 曲线与标准形状公式，统一到局部坐标；曲线依据物理像素误差自适应细分。
- [ ] 实现闭合/开放路径、cap/join/miter、孔洞、填充规则、绕向、退化和镜像处理。
- [ ] 凹路径不能调用 `_concave_indices` 或创建 ImGui scratch draw list。
- [ ] 如需独立剖分依赖，先按 `3rdparty/README.md` 核对许可、固定版本、CPU-only 构建和实际功能；不要自己拼一个仅支持凸多边形的“通用剖分器”。
- [ ] 验证面积、边界、孔洞、自交填充结果和变换；用独立高采样 CPU 参考验证覆盖，不以新实现自己生成的金图证明自己。

参考测试：`test_curves2d.py`、`test_draw2d.py`、`test_ui_icon_concepts.py`。
完成证据：与 UI 无关的可复用网格/覆盖资源，内部三角边没有 AA fringe。

### T06：整体覆盖率、透明组与裁剪（R04/R05；依赖 T05）

- [ ] 区分同色并集、单次绘制透明度、组透明度，明确图层与资源边界。
- [ ] 实现纯色图形的统一覆盖；复杂多色组使用有界离屏池。不要把部件 alpha 连乘伪装成组 opacity。
- [ ] 统一预乘 alpha 和颜色编码的输入边界，定义对 ImGui 颜色的转换方式。
- [ ] 测试 alpha 为 0/0.25/0.5/0.75/1，含重叠、相接、孔洞和嵌套组；检查 RGB 也检查 alpha。
- [ ] Clip 使用实际窗口/控件范围；为描边和 AA 扩大工作边界，橙色参考圆绝不能参与裁剪。

完成证据：同色重叠区域与非重叠区域的期望透明度一致；不同实例重叠仍保持正常前后合成。

### T07：GPU 数据包与 OpenGL 首条完整路径（R01/R07/R08；依赖 T06）

- [ ] 在中立契约中定义几何资源、实例、paint、clip、layer 和有序命令；首版仅支持需要的管线类型。
- [ ] 实现 OpenGL 离屏执行器，复用已有设备与目标；自研路径真正绕过 ImGui 生成和提交。
- [ ] CPU/GPU 缓冲有明确容量、增长与回收策略；冻结帧持有资源引用，完成后才能复用。
- [ ] 简单圆/环/胶囊采用解析快速路径，复杂轮廓走共享网格；不能各自改变 stroke/gap 定义。
- [ ] 补 GPU shader 和资源生命周期测试：重复提交、缩放、目标 resize、销毁、空帧、超限。

建议新增：`python/tests/gpu/test_draw2d.py`。先形成一条可验收的完整离屏链路，再优化上传。

### T08：ImGui 穿插合成桥接（R06；依赖 T07）

- [ ] 在 UI 边界记录自定义绘图片段，通过带类型的标记保留其在 ImGui 通道中的位置。
- [ ] 通道合并/窗口排序后解析标记，交给统一合成器；不在渲染线程执行 Python 回调。
- [ ] 修正导入器对零元素标记、状态重置和不支持回调的行为，不能静默跳过。
- [ ] 测试子窗口裁剪、滚动、浮动面板、弹窗和模态遮挡；保留纹理注册/释放。
- [ ] 检查输入归属：点击、拖动越界、键盘导航和文本编辑不能因替换绘图而失效。

参考测试：`python/tests/gpu/test_ui_interaction.py`、`test_ui_layout_input.py`、`test_ui_corner_controls.py`。
完成证据：生产胶囊能按正确顺序显示在实际 Viewer，相关交互通过；离屏独立模式仍无需 ImGui。

### T09：WebGPU 与 bgfx 执行器（R02/R08；依赖 T07，最终验收依赖 T08）

- [ ] 实现相同资源与命令语义；Python 两个 GPU 后端与 C++ bgfx 共用 CPU 编译结果。
- [ ] bgfx 接入已有 `RenderRuntime`，使用顺序 UI view；新增独立 pass 实现，避免把每种形状继续塞进 Renderer 巨型函数。
- [ ] 明确纹理原点、scissor、framebuffer scale、目标颜色编码和 alpha 混合，不复制一套坐标修补公式。
- [ ] 运行同一组独立绘图、透明合成、窗口穿插及生命周期案例，记录实际后端与 GPU。

参考：`cpp/tests/Contract.cpp`、`RenderRuntime.cpp`、`python/binding_tests/test_native_render.py`、`test_native_viewer.py`。
完成证据：三个后端都通过正确性矩阵；缺少设备的项明确标为未验证，不能算完成。

### T10：常驻资源、实例化和帧预算优化（R07/R08/R13；依赖 T08/T09）

- [ ] 改用局部几何缓存与设备级常驻资源；位置/颜色不能进入几何键，stroke/路径/必要精度必须进入。
- [ ] 实现连续实例数组和 `draw_instances`，再补 `draw_batch` 的跨语言批量路径；始终保序。
- [ ] 预热可见图标组和管线；尺寸/样式变化的编译任务按版本合并，不发布过期结果。
- [ ] 缓存同时限制字节数与条目数，使用帧租约和完成状态保护资源；内存压力测试覆盖淘汰与重建。
- [ ] 根据第 6 节的测量逐项优化，每次记录瓶颈、修改、结果，不能只把热循环换语言便声称提速。
- [ ] 将有类型、有界的 `Draw2DStats` 接入现有操作目录和 CLI 查询，复用截图接口；测试 schema、禁用统计和资源关闭后的行为。

完成证据：移动/着色测试的几何编译和上传计数为零；冷路径、压力路径无漏画，尾延迟不因排队恶化。

### T11：文本与发布能力（R14；依赖 T04/T09）

- [ ] 图形模块与文本服务分开；复用字体来源，明确字体度量、基线、塑形、fallback 和字形图集生命周期。
- [ ] 临时 ImGui 字体导入路径只留在 UI 适配层；独立 text rendering 必须由独立字体提供者完成。
- [ ] 若首版仅交付图形子集，能力表和报告明确 text 尚未独立，不能勾选“全部自定义绘图已脱离 ImGui”。
- [ ] 准备 CPU-only 与 GPU 平台构建检查；保留 Python 公共 API 兼容，缺少组件明确失败或由用户显式选择 legacy。

完成证据：无 ImGui 的导入和离屏图形测试；文本迁移完成后增加中英文、fallback、缩放基线图。

### T12：生产 UI 与可行性工具迁移（R03/R06；依赖 T08/T09/T10）

- [ ] 优先迁移 Icon Library、viewport tools/playback/context hints，再迁移面板图标、视图立方体及其他自绘。
- [ ] 调参导出、生产预览开关与正式 UI 共用同一绘图定义和参数，不保留一套仅用于展示的 painter。
- [ ] 为 UI Feasibility 增加真正的窗口后端选择：当前直接创建 `Window`，传 `BACKEND=bgfx` 不代表切换后端。
- [ ] 保持 `status-info/warning/error`、原鼠标图形、pause 等已确认设计；新旧截图差异逐项说明，不擅自“优化造型”。
- [ ] 按调用点清理已无消费者的旧缓存和 ImGui 特定绘图补丁；原生控件仍需要的部分保留。

完成证据：参数→Icon Library→生产 UI 一致；小尺寸和放大图均已人工检查，截图记录后端来源。

### T13：完整验收和最终交付（所有 R；依赖相关前序）

- [ ] 完成第 5、6 节的适用检查，并自行打开图像与差异图，不只看命令退出码。
- [ ] 用 requirement ID 逐项记录实现和证据；任何未实现/未验证部分列为剩余工作。
- [ ] 更新架构图、公共 API、CLI/RPC 文档和本清单的实际文件路径，删除不再成立的提案描述。
- [ ] 汇总性能提升与退化，保留原始数据、异常和环境信息；禁止仅展示最好的一次。

最终交付至少包含：代码差异、需求覆盖表、测试结果、实际 UI 截图、性能对比、内存/队列限制及遗留项。

## 5. 测试与既有验收方式

### 5.1 使用现有入口

下面均为当前存在的命令。从仓库根目录运行；改动后的适用门禁以
[验证矩阵](../guides/testing.md#change-mapping)为准，不把本文列表当作免除其他检查的依据。

```bash
TMPDIR=/private/tmp make check
uv run --no-sync pytest -q -m 'not gpu' python/tests/test_operations.py python/tests/test_rpc_scene.py python/tests/test_scale.py python/tests/test_pending_model_edits.py
uv run --no-sync pytest -q -m 'not gpu' python/tests/test_rpc_lifecycle.py python/tests/test_rpc_client_validation.py python/tests/test_control_cli.py python/tests/test_cli_automation.py
make agent-control
make agent-viewer
```

纯 CPU 几何先跑 `test_curves2d.py`、`test_draw2d.py` 和新增契约测试；原生实现/绑定使用：

```bash
make cpp-test
make cpp-python-test
make cpp-python-gpu
make native-composition
make native-ui-parity
```

新增的 C++/绑定测试文件必须注册到 CMake/Make 实际执行列表，不能只创建文件而没有被上述目标收集。
涉及绑定机制的改变，还需按验证矩阵执行 `make native-bindings-test`。

GPU 绘图或 shader 修改的适用门禁：

```bash
make gpu
make gpu-wgpu
make gpu-bgfx
```

新增 GPU 文件也要进入 `GPU_WGPU_FILES`，否则默认 WebGPU 目标不会执行它。
单文件迭代可使用 `uv run --no-sync pytest -q -m 'gpu or physics' python/tests/gpu/test_draw2d.py`，
该文件需在 T07 创建后才能运行；不同后端分别设置 `MOJIVE_RENDERER`，bgfx 还需已构建的 `MOJIVE_NATIVE_BUILD`。

涉及 MuJoCo 编辑时：

```bash
make test-physics
make mujoco-audit
make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables
```

### 5.2 必须查看的视觉结果

```bash
make ui-icon-concepts
make ui-layout-audit
make ui-runtime ARGS='--scale-only -o output/draw2d/scale-ui'
make ui-feasibility ARGS='--ui-scale 1.5 --page geometry'
make native-ui-parity
```

现有图标 gallery 首先用于造型和布局基线，不等于三后端全部验证。
检查 hierarchy 三角形/文字高度、snapshot/camera/light 对齐、playback 与 keyframe transport 的独立参数、
rotate 外环/内环/gap、半透明 hidden/light 图标和原鼠标图形。
图标数值以当前 `ui/icons.py` 为准，不从旧聊天截图重新猜测。

新自研绘图需要补充透明背景、多种纯色背景、同色并集、多色组、嵌套裁剪、亚像素移动的自动 gallery。
参考图中已知的 ImGui 合成错误不能当作新实现必须复刻的结果，应另列预期差异，并使用独立参考证明修复。
不以修改 golden 或放宽阈值消除无法解释的差异。

验收惯例：生成结果→agent 自己打开截图/差异图→解释或修复差异→完成适用门禁→交付绝对路径链接。
golden 更新与比较分开执行，具体规则见[视觉基线流程](../guides/testing.md#visual-review-and-baselines)。

### 5.3 环境陷阱

- macOS 默认临时路径过长会触发 `AF_UNIX path too long`；使用 `TMPDIR=/private/tmp`，不要因此重写 RPC。
- `uv sync`、同步式 `uv run`、`make docs-check` 可能用上游 wheel 覆盖项目定制 ImGui；之后执行 `make setup-imgui` 恢复。
- 原生 wheel 缺少 `add_indexed_fill` 等方法时，先核对安装；相关测试被 skip 不是验收通过。
- GPU、窗口测试串行；性能测量单独运行，不能与编译、测试、其他 benchmark 同时争抢资源。
- 纯文字改动只需文档门禁，无需为段落或命名写测试。正常代码改动完成后运行 `make check`。
- 设备/显示/依赖阻塞先尝试恢复，仍不可用时记录具体命令和覆盖缺口，不能标记通过。

## 6. 如何与 ImGui draw list 比较性能

### 6.1 基线必须公平

旧路径必须是“当前生产几何 + 上一轮缓存 + 项目定制 ImGui wheel”，不能退回未优化的裸 Python 逐顶点路径。
新旧使用相同图标定义、stroke/padding/gap、颜色、透明度、裁剪、DPI、窗口与 viewport 像素尺寸。
先比较输出，再比较速度；通过少画部件、改曲线精度、关闭 AA 或去掉透明组得到的速度不可接受。

分别比较三层，不能把不同范围的数字混在一起：

| 层次 | 旧路径 | 新路径 | 报告范围 |
| --- | --- | --- | --- |
| CPU 几何/记录 | 生产 painter → ImguiDraw2D，使用真实 draw list | 相同 painter/图形 → 自研记录与编译 | 编译、记录、命令数、分配；不代表 GPU 速度。 |
| 同后端绘图 | 同一 GPU 后端提交 ImGui 生成的数据 | 同一后端提交 owned 数据 | 上传、CPU 提交、GPU 时间、绘制批次、完成帧。 |
| 完整 UI | 旧绘图的实际 Viewer | 新绘图的实际 Viewer | 帧耗时、交互尾延迟、输入响应、整体内存；场景与物理状态保持一致。 |

### 6.2 现有性能入口

```bash
make g3-benchmark ARGS='--batches 7 --profile -o output/draw2d/baseline/g3.json'
make interaction-benchmark ARGS='--batches 7 --profile -o output/draw2d/baseline/interaction.json'
MOJIVE_RENDERER=opengl make ui-frame-profile ARGS='--frames 300 --warmup 30 --profile-frames 180 --expand-hierarchy --diagnostics -o output/draw2d/baseline/opengl'
MOJIVE_RENDERER=wgpu make ui-frame-profile ARGS='--frames 300 --warmup 30 --profile-frames 180 --expand-hierarchy --diagnostics -o output/draw2d/baseline/wgpu'
make native-python-build
MOJIVE_RENDERER=bgfx MOJIVE_NATIVE_BUILD="$PWD/output/cpp-build" make ui-frame-profile ARGS='--frames 300 --warmup 30 --profile-frames 180 --expand-hierarchy --diagnostics -o output/draw2d/baseline/bgfx'
```

`output/cpp-build` 是当前默认构建目录；若改了 `NATIVE_BUILD`，必须同步设置扩展路径。
`ui_frame_profile.py` 已有 `--keyframes`、`--pending-edits`、`--hover-gizmo --gizmo-mode rotate`，
分别加入对应工作负载。现有 `tools/benchmark_g3.py` 区分实际 ImGui 提交与 NullDraw，不能拿 NullDraw 当绘图结果。

### 6.3 要新增的对照能力

在现有 `ui_frame_profile.py` 与 `tools/benchmark_g3.py` 中扩展明确的选择参数，例如
`--draw2d-backend imgui|owned`，并增加 `--compare-draw2d`。
**这些参数目前不存在，必须实现后才能调用。** 选择项只在 composition root 生效，不散落成各控件的分支。
报告必须包含 `requested_draw2d`、`effective_draw2d`、`requested_renderer`、`effective_renderer`。
实际路径不匹配时立即失败，避免“两个选项实际都在画 ImGui”。

同进程交替测量建议采用固定 AB/BA 顺序、至少 5 轮，每轮各 300 帧，先预热各 30 帧。
如果设备生命周期不允许同进程切换，采用独立进程交替、记录不可比因素；不能隐式重置一方的缓存而保留另一方缓存。
冷路径使用新记录器/资源缓存，热路径预热后测量；首次编译和上传必须单列，不包含在热路径里后又声称不存在。

必测工作负载：

- 静止：24、256、2048 个图标，覆盖多图标族，而不仅重复一个矩形。
- 移动/旋转/颜色变化：几何不变，检查编译与几何上传计数。
- 尺寸、stroke、gap 连续变化：检查重新编译、缓存命中、任务合并和过期结果。
- 滚动、大量裁剪、弹窗遮挡、嵌套透明组：覆盖顺序和图层成本。
- RPC 突发编辑与 Scale 拖动：单独报告队列等待、命令完成时间和最长 UI 帧。

不要以 Python 函数返回当作 GPU 完成。GPU 时间使用后端时间戳与提交编号，延迟读取，不每个 draw 强制同步。
吞吐测试在整轮末尾等待完成；readback 成本单列。计时阶段避免截图、日志和 cProfile；另跑一轮采样分析。

### 6.4 报告字段与验收原则

至少保存：实现版本/工作区校验值、环境、参数、原始每帧样本、p50/p95/p99、编译/记录/上传/提交时间、
GPU 时间、draw call/批次数、几何与实例上传字节、缓存命中/淘汰/字节数、离屏图层数量/面积、RPC 等待、
RSS 峰值与关闭后的资源计数。GPU timer 不可用记为 null，不记为 0。

硬性功能指标：静态几何在仅移动/颜色变化且没有缓存淘汰的热路径中，重新编译和重新上传次数为零；
资源增长受预算约束；没有丢帧内容或失效资源；输入/透明/裁剪正确。
时间指标先沿用已有 `ui-frame-profile` 的胶囊 CPU 预算 0.75 ms，保持相同测量范围。
同时比较新旧 p95/p99，多轮稳定退化必须解释并修复；不要拿跨机器绝对数值制定通用硬阈值。

历史数字仅供定位，不能代替新实验：上一轮同窗口胶囊 CPU 中位数从 1.516 ms 到 0.318 ms；
2000 对象的尺寸预览从 5.276 ms 到 0.011 ms。两项不是新 Draw2D 的成果。
此前不同运行的整帧中位数反而是 5.17 ms 与 8.02 ms，受 GPU/窗口等待影响，不能宣称 FPS 提升。
本地原始证据在 `output/scale-cleanup/`；该目录可能不随仓库分发，缺失时重新测量。

## 7. Mojive 中常见的优化入口与方法

| 瓶颈表现 | 优先检查 | 合理做法 |
| --- | --- | --- |
| 静态图标仍消耗大量 Python CPU | `icons.py`、`draw2d.py`、`curves2d.py` | 拆开几何与放置；缓存真实不变输入；将批量转换一次完成。 |
| 拖动时越来越卡 | 缓存键、任务队列、临时图层 | 排除鼠标位置造成的无限缓存；有界 LRU/内存预算；合并旧版本编译任务。 |
| UI 提交复制太多 | `window_native._draw_packet`、绑定 `render_ui`、bgfx transient buffer | 常驻几何与实例更新；减少边界复制，但保留明确数据所有权。 |
| 改一个尺寸复制整个场景 | `ModelEditDraft.refresh_preview`、`GeometryPreview` | 同拓扑复用数组和 node index；结构变更才重建；不要破坏现有优化。 |
| 一批编辑重复计算物理 | `Session.apply_model_edits`、适配器 `model_edit_batch` | 统一编辑计划、合并常量更新与重建，保留最终一次刷新和失败回滚。 |
| 大 hierarchy 或长表格卡 | hierarchy panel、布局与文本测量 | 只画可见行；按文本、字体、字号与版本缓存度量；保留键盘和滚动选择语义。 |
| 调试线/箭头过多 | `render/debugdraw.py`、各后端 debug pass | 复用 packed buffer 与批量数组；共享世界/屏幕几何，保留 occlusion 和单位区别。 |
| 帧时间主要在等待 | GPU 时间戳、readback、present、队列 | 区分实际 GPU 工作与同步，不用更多 Python 缓存掩盖 GPU 等待。 |

优化循环：保存基线→用 profile 定位→提出一个具体成本假设→修改最小职责范围→验证图像/行为→重新测量。
避免重复重构已无热点的代码；不要把所有 if/函数合并进一个更大的 dispatcher 来追求行数少。

## 8. 每阶段交接格式

每一阶段完成后在 `output/draw2d/` 保存记录，并更新本清单对应项：

```text
任务：Txx；需求：Rxx
实际修改：文件、关键符号、接口与所有权变化
实现状态：完成 / 部分完成 / 未开始
测试：真实命令、退出结果、失败与恢复过程
视觉：图像/差异图路径、已检查的现象
性能：同条件 before/after、原始报告、是否有退化
兼容性：调用方迁移、旧路径、未验证平台
下一步：下一个具体任务及其依赖
```

可直接给接手 agent 的任务说明：

> 请从根目录 AGENTS.md、docs/plans/draw2d-implementation.md 和 docs/plans/draw2d-architecture.md 开始。
> 从包含本交接文档及 `9214213` 的分支开始，保留已提交的 Scale、图标参数与缓存优化，以及接手时存在的其他工作区改动；先完成 T00，再按依赖逐项实现。
> 本文包含需求、源码导航、接口约束、测试和性能对比办法，请据此执行，不要只再输出一份计划。
> 每阶段自行检查图像和实际行为，记录证据后更新任务清单；未完成、未测试的项目如实列出。
> 不需要创建新的 agent 或新的 Codex 任务；先在当前任务完成授权工作。
