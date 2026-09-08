# Mojive 原生 C++ 方案与 bgfx 评估

> 初始评估日期：2026-09-08。以下保留评估时的计划与判断。
> 后续已在此分支实现首轮原生验证；当前范围、接口与复现入口见[原生验证指南](../guides/native-probe.md)。
> Mojive 基线：`origin/main` 的 `23194d361dbaed19448cb5512af42e9936452805`。
> bgfx 源码核对版本：`9b636df330c81e11c84595651a291b6c59fb7396`，提交时间为 2026-09-07 UTC。
> 该 SHA 用于固定本次证据，不代表已验证的生产依赖版本。

## 后续实验更新

**当前选型与验收状态以 [C++ 渲染后端选型结论](native-backend-decision.zh.md) 为准。** 决定使用 bgfx 启动 C++ 生产实现，nanobind 用作 Python 薄接口，SDL GPU 保留为可替换接口的实验对照。已补充原生 MuJoCo 并行、跨 pass 依赖、窗口生命周期、可移植 shader 和 ASan/UBSan 验证。

用户已决定暂不运行 workflow CI，Linux 后续在对应系统手工验证；物理 GPU、实际高 DPI 和完整生产画质仍是发布与迁移验收条件。以下保留初始提案，不能将早期候选措辞或旧 benchmark 当作最新结论。

## 先看结论

**建议把 bgfx 作为原生 C++ 版本的首选原型候选，先通过 Mojive 的关键能力和性能验证，再决定正式采用。** 它符合桌面三平台、原生图形 API、Dear ImGui、自定义渲染流程、避免引入大型引擎的方向。bgfx 使用 BSD-2-Clause 许可证；本方案排除 Diligent，不采用 Qt，也不引入 Dawn 作为原生渲染依赖。[官方概述](https://bkaradzic.github.io/bgfx/overview.html)、[许可证](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/LICENSE)

**长期目标是 C++ 核心、C++ 编辑器与 Python SDK。** 渲染、物理运行时、逐帧场景处理、Session 和 ImGui 面板最终都可在没有 Python 解释器的原生应用中运行。Python 实现继续保留，承担现有生产使用、功能对照、快速试验和自动化；迁移期间不整体替换当前包。

最值得先确认的三件事是：

1. bgfx 的 GPU 回读会不会妨碍低延迟拾取和连续录制。这是本次发现的首要性能风险。
2. 整数 object ID、语义分割和米制深度能否跨后端保持现有精度及含义。
3. C++ 场景处理加原生 Metal，相比**目前已启用物理线程的 main**，还有多少实际收益。

“用了 C++、bgfx、多线程”本身不算验收结果。已经 GPU 受限或 MuJoCo 原生步进受限的场景，不会因为换语言就自动变快。

## bgfx 是否适合 Mojive

| 需求 | 判断 | 对 Mojive 的具体影响 |
|---|---|---|
| macOS、Windows、Linux | 合适 | 首选 Metal、D3D12、Vulkan；Windows D3D11 可作为后续兼容后端 |
| 多线程渲染提交 | 合适，有规则 | API 与渲染执行可以并行，多个 encoder 可以录制命令；应用仍需安排物理、场景更新及资源任务 |
| 自定义 renderer | 合适 | 可以自行组织阴影、透明物体、轮廓、辅助图形和离屏输出；这些功能需要迁移 Mojive 的实现 |
| 完整现代显式 RHI | 有取舍 | 它对资源状态和提交做了抽象；若以后要求精细控制多个 GPU 队列和跨队列同步，应重新评估适配性 |
| GPU picking、连续图像导出 | 必须先实测 | 异步返回不等于 GPU 无等待，不能直接照搬当前同步回读路径 |
| 多相机、多个 SceneRenderer | 可以设计支持 | 共用一个进程级 bgfx runtime，各自拥有 render target，集中推进帧 |
| ImGui docking 和独立系统窗口 | 需要集成 | 官方示例可参考，但不是可以原样接入 Mojive 的完整 docking backend |
| CJK、HiDPI 和自有设计规范 | 可实现 | 属于 ImGui、字体及 Mojive 组件层；bgfx 不负责布局，也不决定圆角曲线 |
| 维护成本 | 比三个自研图形后端低，但并非零 | 固定依赖、迁移 shader、维护 ImGui renderer backend，仍需三平台实机验证 |

平台覆盖参考[官方概述](https://bkaradzic.github.io/bgfx/overview.html)。线程和 encoder 的约束参考[线程模型](https://bkaradzic.github.io/bgfx/internals.html)。多窗口能力参考[22-windows 示例](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/examples/22-windows/Windows.cpp)。上述“合适”是架构判断，不是已经通过运行测试。

### 1. 最需要警惕的是回读

当前 Mojive 的 `pick()` 返回 object ID，`SceneRenderer.render()` 同步返回 NumPy 图像；bgfx 的纹理回读返回一个完成帧号。核对版本中的 API 名为 `bgfx::read(TextureRegion, ...)`，旧资料常称 `readTexture`。

更关键的是，上游明确提示回读会阻塞 GPU，不适合直接放进主渲染循环。Metal 实现也确实存在提交后等待 command buffer 完成的路径。因此不能把“异步 API”解释为“连续回读不会卡渲染”。[回读接口](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/include/bgfx/bgfx.h#L3584)、[Metal 回读实现](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/src/renderer_mtl.cpp#L1630)、[等待实现](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/src/renderer_mtl.cpp#L5328)

计划采用小尺寸 staging target 和有界回读池。拾取先 GPU 拷贝需要的像素区域，再回读完整 staging mip；轮廓效果留在 GPU。连续录制单独测量全帧回读成本，不能用一次截图的结果替代。

异步结果携带 scene revision、frame ID、相机版本、viewport generation 和输入序号。点击绑定点击时显示的帧，过期结果不能选中后来复用同一 ID 的实体；hover 可以合并旧请求。resize、模型替换和退出必须让待处理请求完成或明确取消。

Python SDK 增加异步入口；原同步入口继续提供相同结果。非 GPU owner 线程可等待并释放 GIL；owner 线程必须受控推进工作，不能等待自己队列中的任务。原生 UI 使用异步路径，避免为截图或拾取停止事件处理。

### 2. ID、深度和 shader 需要按契约迁移

bgfx 的 picking 示例使用颜色编码，不能直接作为 Mojive 完整 uint32 object ID 的实现。应优先验证整数 render target；若使用颜色打包，则必须无损覆盖 32 位，并关闭会改变编码的混合、sRGB 和 MSAA resolve。[示例 shader](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/examples/30-picking/fs_picking_id.sc)

原型必须检查设备格式能力，而不是仅检查枚举存在。测试包含超过 `2^24` 的 object ID、负数分割背景、多个相机和尺寸变化。建议颜色使用 MSAA，ID 使用独立单采样 pass，沿用目前对整数拾取边缘的明确处理。

公开输出保持：RGB 为 `uint8[H,W,3]`，米制深度为 `float32[H,W]`，object ID 为 `uint32[H,W]`，语义分割为 `int32[H,W,2]`。保留 `out`、背景值、坐标方向和源元数据的含义；object ID 与物理 body index 不混用。

shader 采用 bgfx shaderc 的离线构建流程。现有 GLSL 不能假定原样跨平台；投影深度范围、纹理上下方向、整数输入、裁剪和透明排序都要单独核对。OpenGL 的 geometry shader 或宽线依赖应改为可移植的三角形展开等实现。[shader 工具](https://bkaradzic.github.io/bgfx/tools.html)

### 3. bgfx runtime 必须集中管理

核对版本使用进程级 context。因此不能为每一个 `SceneRenderer` 调用一次 `bgfx::init()`，也不能由一个 peer 关闭共享设备。[context 实现](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/src/bgfx.cpp#L397)

设计一个 `NativeRenderRuntime`，拥有设备、提交时钟和资源回收；多个 viewport、相机和离屏 renderer 使用独立 target、共享 runtime。所有 `frame()` 由一个 owner 推进。独立输出任务采用公平的队列和预算，不能让后台批量渲染饿死交互 viewport。

另外，当前 OpenGL UI 使用 GL texture ID。bgfx Metal texture 不能直接塞进这个接口。第一阶段原型使用独立原生窗口或离屏输出；最终由同一个 bgfx runtime 渲染场景和 ImGui，避免逐帧 GPU→CPU→GPU 传图。

### 4. ImGui 后端和设计系统是独立工程

核对的 bgfx 示例处理了 ImGui 动态纹理更新，但捆绑的是 master 系列 ImGui，没有完整提供 docking 分支的多 viewport 平台/渲染回调。窗口内拖动浮动面板，与把面板拖到独立系统窗口，是两种验收场景。[ImGui renderer 示例](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/examples/common/imgui/imgui.cpp)、[捆绑的 ImGui 头文件](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/3rdparty/dear-imgui/imgui.h)

建议固定 Dear ImGui docking 版本，采用官方 GLFW platform backend，维护 Mojive 自己的小型 bgfx renderer backend。原生编辑器只拥有一套 ImGui context，避免与 Python `imgui-bundle` 的另一份编译产物交换内部指针。

保留 JetBrains Mono + Noto Sans CJK/SC 字体组合。原生 ImGui 圆角仍为 **4.8 个逻辑像素**，按 DPI 缩放；自绘组件继续使用曲率连续过渡。未来若重新统一底层圆角，C++ 源码构建确实能免去自制 Python wheel，但仍需验证 clipping、菜单、tab、dock 和命中区域。bgfx 本身不能解决这些问题，也不应成为再次扩大圆角补丁的理由。

设计系统独立定义行高、横纵内边距、文字基线、图标视觉中心、badge、表格列和窄面板换行。此前的 status、Hierarchy、Inspector、菜单、Keyframes 和 Assets 问题应成为原生 UI 的回归清单。

## 推荐技术栈

| 部分 | 建议 |
|---|---|
| 原生语言与构建 | C++20、CMake、Ninja；避免把新编译器特性作为初期前提 |
| 渲染 | bgfx + 配套 bx、bimg；先 Metal，再 D3D12/Vulkan |
| 窗口和输入 | GLFW，延续现有窗口体系；平台限制集中在 window 层 |
| UI | Dear ImGui docking + FreeType + Mojive 设计组件 |
| 物理 | MuJoCo C API；其他引擎仍通过 adapter 契约接入 |
| 并发 | 有界任务队列、线程池、取消令牌和明确的状态所有者 |
| Python 绑定 | pybind11，以场景、帧和命令为调用粒度 |
| Python 原生包构建 | scikit-build-core；初期作为可选原生包，不替换现有 Python 构建 |

bgfx 默认构建体系之外，有上游作者维护的 [bgfx.cmake](https://github.com/bkaradzic/bgfx.cmake)。选定一套经过测试的 bgfx/bx/bimg/CMake port 版本，shaderc 与运行库配套固定；不分别追踪几个仓库的最新提交。[构建文档](https://bkaradzic.github.io/bgfx/build.html)

显式禁用 `BGFX_CONFIG_RENDERER_WEBGPU` 及对应构建选项，核对产物依赖中没有 Dawn。只选择 Metal 并不等价于已经从构建配置中排除了 WebGPU。[后端配置](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/src/config.h)

pybind11 进入 C++ 时不会自动放开 GIL。完成参数检查后，对原生运行、等待和批量计算显式释放；调用 Python 回调时重新获取。因此实时物理和渲染任务不能依赖逐实体 Python 回调。[GIL 规则](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)、[原生包构建](https://scikit-build-core.readthedocs.io/en/latest/)

## 哪些放到 C++，哪些继续留在 Python

| 模块 | 最终归属 | 原因和边界 |
|---|---|---|
| 场景契约、object ID、变换与边界计算 | C++ core | 稳定结构与动态帧分离；复用缓冲区，消除逐实体跨语言调用 |
| 可见性、实例分组、动态网格上传、render pass | C++ renderer | 这是目前 Python 逐帧处理和调度的主要迁移对象 |
| GPU 资源、shader、回读、截图与录制提交 | C++ renderer | GPU 生命周期、帧时序和异步结果统一管理 |
| MuJoCo 运行、控制命令、快照发布 | C++ adapter/runtime | 无 Python 热路径；MuJoCo 状态仍由 adapter 独占 |
| Session、选择、Undo/Redo、编辑事务 | C++ session | 最终提供唯一状态源，避免原生 UI 和 Python 分别维护选择 |
| 窗口、ImGui 面板、gizmo 和输入 | C++ editor | 完整原生编辑器不需要解释器，也不逐帧等待 Python |
| AI 工作流、场景编写、批处理和试验性 adapter | Python | 开发效率高，通过稳定 SDK 和帧协议接入 |
| 现有 Python Viewer 和 renderer | 迁移期间继续保留 | 生产回退和行为对照，不随原型一起重写 |

C++ renderer 只依赖公共场景契约，不依赖 ImGui 或 MuJoCo。native library 的离屏使用不要求初始化编辑器 UI。未来 Python SDK 与原生可执行文件复用同一套核心。

跨语言默认采用一次批量提交和 C++ 自有缓冲池；只有生命周期能证明安全时才提供零拷贝。bgfx 的 `makeRef` 允许延迟释放，回调可能发生在其他线程，不能借用任意 NumPy 指针后立即让调用者释放或修改数组，也不能在无 GIL 的 GPU 回收线程销毁 Python 对象。[内存生命周期契约](https://github.com/bkaradzic/bgfx/blob/9b636df330c81e11c84595651a291b6c59fb7396/include/bgfx/bgfx.h)、[NumPy 绑定](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)

MuJoCo C API 与 Python 包也要固定兼容版本。原生自有模型由原生库加载；外部 Python 调用者持有的模型先通过现有 adapter/帧接口接入，不把不同 MuJoCo 库实例的内部指针直接互传。

## 并发如何落地

目前 main 已能让释放 GIL 的 `mj_step` 与 Viewer 工作重叠。C++ 的新增价值，是把场景转换、调度、编辑器和资源工作也移出 Python 热路径，减少对象遍历和临时内存，并提供可控制的任务调度。它不会把一个 MuJoCo 世界自动均匀分配给所有核心；独立世界与单世界内部并行要分别处理。[MuJoCo 多线程说明](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#multi-threading)

| 执行单元 | 职责 | 同步方式 |
|---|---|---|
| OS 主线程 / 编辑器 owner | 窗口事件、ImGui、Session 事务、准备当前 UI 帧 | 输入命令有序提交，不等待资源加载或 Python 回调 |
| bgfx API owner | 设置 view、汇总 encoder、推进一帧 | 常规部署可与编辑器 owner 合并；全进程唯一 |
| bgfx render thread | 执行 GPU 资源命令、绘制和呈现 | 使用 bgfx 的帧交接，不额外叠加一个同职责线程 |
| 物理线程 | 独占模型的可变物理状态、执行步进、应用控制 | 发布完整的一致快照；渲染持有只读租约 |
| 有界工作线程池 | 资源读取、解码、网格预处理、独立场景计算 | 完成后发布结果；失效任务可取消，不直接修改 UI |

OS 主线程职责与 GPU 提交线程不是随意互换的。GLFW 窗口创建和事件处理遵守其主线程限制；bgfx 支持自行驱动 `renderFrame`。三平台原型分别验证线程安排，若平台要求主线程执行 GPU 工作，就采用显式 owner 调度，不靠跨线程调用绕过限制。[GLFW 线程限制](https://www.glfw.org/docs/latest/intro_guide.html#thread_safety)、[bgfx 线程与帧交接](https://bkaradzic.github.io/bgfx/internals.html)

物理与渲染不共享正在写入的 `MjData`。显示采用最新完成快照，保留读者租约的缓冲区不被覆盖；相机和选择无需等待物理步。暂停、单步、重置、Undo/Redo 和模型替换则有明确命令完成边界。模型编译可准备新结果，真正替换时验证版本并协调所有者。

下一步可评估把显示快照缩减为位姿、所需传感器、接触和动态网格等字段；录制、回退和精确状态恢复另有完整状态路径。不能只复制位姿就宣称所有 MuJoCo 功能仍然兼容。外部程序拥有物理时钟时，Mojive 不启动第二个步进线程。

实时显示允许丢弃未消费的旧快照；要求无损的录制队列采用背压并报告延迟。这两个语义必须分开。Python `async` 接口需要连接真实原生任务完成事件，不把阻塞工作换个函数名。CPU 异步任务也不等于 GPU async compute。

线程池按 CPU 数量和 MuJoCo 自身并行设置控制预算，先少量 worker 再测扩展性。增加 encoder 会带来同步和内存成本，bgfx 帧流水也可能增加延迟，因此不能以“核心占满”作为优化目标。

## 实施顺序与通过条件

下面是后续实施计划，本次只完成源码评估和提案。阶段按可验证结果推进，不以未经实测的工期承诺代替风险判断。

| 阶段 | 交付 | 进入下一阶段的条件 |
|---|---|---|
| P0：关键能力原型 | 原生窗口、最小 ImGui docking、四类离屏输出、拾取和回读测试、设备能力报告 | 本机 Metal 正确运行；回读成本可接受；没有依赖 Dawn；记录未验证的平台 |
| P1：静态/动态场景 renderer | 从相同场景快照渲染，迁移必要材质、阴影、轮廓、辅助图形和动态网格；提供粗粒度 Python 入口 | 输出契约、视觉对照、多 renderer 生命周期和资源更新通过；得到原生渲染实测结果 |
| P2：原生物理及帧管线 | MuJoCo C API、快照池、控制队列、错误恢复和关闭 | 相同步数轨迹一致；100 humanoid、布料、暂停/单步/回退/重载通过；延迟及内存无不可接受退化 |
| P3：原生 Session 与完整编辑器 | 成对迁移编辑状态与各面板、gizmo、快捷键、菜单、资源浏览和工作区 | UI 状态只有一个来源；选择/focus/Undo/Redo 一致；窄面板、CJK、HiDPI、docking 视觉和输入测试通过 |
| P4：SDK 与兼容 | Python SDK、异步 API、远程输入、场景格式、录制及导出兼容 | 现有关键 Python 用例可运行；同步结果含义不变；外部时钟和回调边界明确 |
| P5：三平台交付 | macOS、Windows、Linux 原生应用及 Python wheels | 实机 GPU 验证、打包和退出/异常测试完成，再讨论默认切换 |

P0 必须包括 32 位 ID、带负数的分割结果、米制深度、MSAA 下的拾取、1×1 和全帧回读、连续 resize、HiDPI 以及多个输出 target 的创建/关闭。独立系统窗口另测 swap chain 支持与窗口生命周期；无显示服务器的 Linux 渲染也单独验证，不能用隐藏窗口冒充 headless 支持。

P1 可先复用现有场景快照，避免在验证图形后端时同时重写所有面板。P3 不长期维护两套相互镜像的 Session；完成一条编辑链路后，Python 侧通过绑定访问同一状态。原型不直接写进 Python OpenGL 窗口。

已增加 `native-probe`、`native-test`、`native-benchmark` 和 `native-gallery` 等 Make 入口，并接入现有验证矩阵。当前本机结果见[原生验证指南](../guides/native-probe.md#initial-measured-result)。

若关键回读只能靠深度修改多个 bgfx 后端才能满足交互与录制要求，或者原生帧延迟持续明显退化，就暂停扩大迁移范围，重新审视 bgfx。不能因为已经写了几个 pass 就把它当成不可撤回的选择。

## 怎么判断有没有性能收益

已有 main 的本机测试中，100 humanoid 从串行约 63.9 FPS 到并行约 119.9 FPS，达到测试的 120 FPS 应用上限；物理保持约 200 步/秒。**这是之前 Python 并发改造的结果，不是 bgfx/C++ 数据。** 该组实际 viewport 为 696×518，不能称为完整 1080p 渲染表现。场景有 1,600 个运动刚体、2,700 个自由度和 1,901 个 geom；三个完整快照约占 310 MiB 预留容量，不能直接等同于 RSS。[现有验证流程](../guides/testing.md#renderer-performance)

新的对比基线是同一 `origin/main` 的默认 threaded physics 模式。测试分两组：固定轨迹的 renderer replay 隔离图形成本；实际交互编辑器测试同时记录物理进度和状态延迟。必要时用相同后端的原生对照或分段 CPU profile，区分语言迁移与 OpenGL→Metal 的收益。

建议每项预热后测试至少 30 秒、交替运行五次，记录中位数及 P95/P99，并保留原始时间线。基准矩阵至少包括：

- 轻场景、100 humanoid、布料/动态网格、密集静态网格、大量 debug 图形和复杂面板。
- 明确的物理 framebuffer 尺寸：1920×1080、2560×1440，以及常规编辑器布局；一致的 MSAA、阴影、反射和 ID pass。
- 不限帧吞吐与 60/120 FPS 节奏下的交互延迟分开测试；显示器刷新率、VSync 和 DPI 记录在报告中。
- GPU pass 时间、主线程/提交 CPU 时间、物理步数每秒、已显示状态落后、队列等待、回读时延、上传字节和 RSS。
- 连续拖动物体/相机、选择/focus、resize、模型重载、全帧录制和取消任务，检查卡顿与错误结果。

通过条件先看正确性，再看多次运行可重复的收益。对轻场景尤其检查延迟是否增加，不能用重场景平均 FPS 掩盖。测得 CPU 提交完成或 `present` 返回也不是显示器发光延迟；需要声明测量边界。

现已完成本机 Metal 原型和固定轨迹测量；完整画质对比、真实 2× framebuffer 及 Windows/Linux 实机验证仍待完成。当前不能据此给出“值得全面迁移到 bgfx”的性能结论。

## 分支与保留 Python 的方式

已从获取到的最新 `origin/main` 创建 `codex/bgfx-cpp-evaluation`，独立 worktree 为 `mojive-bgfx-cpp`。原 main 工作区保持不动。该分支已承载提案和首轮本机原生验证；更大阶段按完整行为拆成可审查的提交或分支。

原生代码建议新增 `cpp/` 与独立绑定包；现有 `python/src/mojive/` 继续运行。优先保持场景/帧/命令和公开输出契约兼容，不靠大范围重命名制造两套相似接口。达到对应验收条件后再切换入口，Python 版本的删除或默认后端变更不包含在本次提案工作中。
