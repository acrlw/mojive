# C++ 渲染后端选型结论

> 2026-09-08；Apple M5 / macOS / Metal。本文决定启动阶段的技术栈，不宣称原生版已达到生产 Python 版的全部功能与画质。

**选择 bgfx 作为 Mojive C++ 部分的默认渲染后端，C++20 + Dear ImGui + nanobind 作为核心组合。** 保留后端中立接口和 SDL GPU 实验适配器；后续集中实现一套生产渲染功能，不继续平行维护两套原生产品。Python 版继续保留。

这是当前交互式编辑器目标下的工程选择。SDL GPU 的拾取完成更快，接口也更直接；bgfx 当前实现的主线程余量更大，现有线程、shader 和诊断设施更符合本阶段减少平台工程量的要求。不要把这一结论解释成 bgfx 在所有场景都更快，或 SDL 不能做多线程。

Windows/Linux CI 已准备，但当前 GitHub OAuth 凭据缺少 `workflow` scope，远端拒绝新增 workflow，因此 **这两个平台的构建和运行未通过本轮 CI 验证**。本机生成了 SPIR-V、MSL、HLSL；HLSL 生成不等于 Windows DXIL 编译或 D3D12 运行通过。真正的 Windows/Linux GPU、Linux Wayland、2× framebuffer 和跨显示器 DPI 也尚未验收。选择默认实现可以推进；三平台发布不能越过这些门槛。

## 验证覆盖

| 项目 | 结果与边界 |
|---|---|
| 可替换后端 | 两个适配器使用同一个标准库 `Renderer`、场景、相机、输出和 ticket 契约；公共核心不依赖 bgfx、SDL、ImGui、MuJoCo 或绑定库 |
| 输出正确性 | 32 位 object ID、两个有符号分割值、米制深度、背景值、4× MSAA、区域拾取、版本与帧来源检查通过 |
| 数据生命周期 | resize/reload/destroy 取消、队列上限、旧快照保留、动态 mesh、错误线程拒绝、runtime 重启通过 |
| 多 pass | 纹理采样、上下方向、alpha 混合、scissor、与资源创建顺序相反的 pass 依赖通过；修复了 bgfx 适配层原有的 view 排序错误 |
| 资源循环 | 每个后端完成 120 次离屏资源创建、采样、读回、缩放或销毁循环 |
| 窗口 | 每个后端完成 24 次 peer 重开、原地 surface 缩放、最小化和恢复；关闭 peer 后其余输出继续有效 |
| ImGui | 原有字体、动态 CJK atlas、docking、主窗口与 peer 缩放；仍是集成 fixture，不是完整编辑器 detached-window 回调实现 |
| 真实物理 | C++ 直接调用 MuJoCo C API，100 humanoid；120 帧原生姿态先与 Python 导出核对，再比较串行与独立物理线程 |
| 内存检查 | 自有 native 代码通过 ASan/UBSan；没有仪器化全部第三方库，也没有宣称通过 leak sanitizer |
| 跨平台 shader | SDL 从同一套 GLSL 离线生成 SPIR-V / MSL / HLSL，生成的 Metal shader 在本机通过 GPU 检查；Windows DXIL 编译仍受上述 CI 权限限制 |

## 性能结论的含义

最终比较使用同一个 100 humanoid 模型、1080p、4× MSAA、120 FPS 软件节拍和相同的输出调度规则。这是离屏渲染测试，不包含完整编辑器 UI，也不代表显示器实际呈现 120 Hz。物理线程独占 `mjModel/mjData`，通过三个预分配且所有权明确的快照缓冲交换姿态。串行组和并行组均实际步进，不使用录制帧代替实时物理。

拾取组每帧请求一个像素；连续输出组每两帧拾取一次、每四帧请求一组 RGB、深度、分割输出。在 120 FPS 时分别为 60 Hz 和 30 Hz；串行组帧率下降时，有效输出频率也随之下降，不能宣称这些低帧率组完成了同样多的输出。多相机组另加一个 640×480 视图及其每四帧一次的 RGB 输出。队列满时等待并计入耗时，最终对比不丢弃请求。每组交替测试三次、每次 10 秒，保留各次原始结果。

| 并行物理，1080p / 4× MSAA | bgfx | SDL GPU |
|---|---:|---:|
| 仅拾取：主线程 P95 | 0.11 ms | 0.27 ms |
| 拾取 + 三种连续图像：主线程 P95 | 3.19 ms | 12.28 ms |
| 再增加第二相机：主线程 P95 | 3.34 ms | 12.68 ms |
| 仅拾取：读回完成 P95 | 17.76 ms | 8.27 ms |
| 第二相机组：读回状态年龄 P95 | 23.04 ms | 25.06 ms |

两种后端的所有并行组都约为 **120 FPS、200 物理步/秒**；连续输出和双相机组每次运行均完成 600 次拾取与 300 组 RGB/深度/分割请求，另一个相机的输出也完整消费。

串行拾取组主线程 P95 分别为 11.16 / 11.24 ms；拆出物理线程后降到表中的 0.11 / 0.27 ms。这个收益表示主线程不再等待物理步进，不是 MuJoCo 求解器本身突然快了数十倍。本轮也没有启用 MuJoCo 求解器内部并行。

串行双相机组出现退化和波动：bgfx 三次为 102.0–120.0 FPS，中位数 114.4；SDL 为 113.5–115.5 FPS，中位数 114.0。该结果同样保留。最终推荐使用已经验证的独立物理线程结构，不依据串行数据宣称 bgfx 全面领先。

帧 CPU 耗时包括构建、提交、poll 解码和队列等待，不包括软件节拍的 sleep。快照年龄从发布时刻起算；读回延迟从进入请求 API 前到应用观察到完成，不是 GPU 独占执行时间，也不是鼠标输入到屏幕发光的延迟。SDL 的 portable GPU timestamp 数据在当前接口中不可用，不能把空值当成零。主线程 P95 更低不等于进程总 CPU 或功耗更低：bgfx 的部分工作发生在内部渲染线程，SDL 也可以通过额外的线程调度减少 UI 线程负担，本轮比较的是已经实现的两种适配器。

现有 Python 版已经能让 MuJoCo 原生调用和渲染并行。这里证明的是 C++ 热路径与所有权设计可用，并比较两个原生后端的负担。没有把简化 native FPS 除以完整 Python 编辑器 FPS，也没有把 nanobind 的小调用收益当成整个应用的加速比。

## 为什么选 bgfx

1. **先保证编辑器主线程余量。** 平均 FPS 到达上限之后，持续输出时的长帧、状态年龄和 UI 处理余量更有区分度。本轮选型以这些数据为依据。
2. **已有内部渲染线程、encoder 和配套 shaderc。** bgfx 的 API 提交与内部渲染可形成流水；跨平台 shader 与运行库一起固定。SDL 可以多线程录制命令，但交换链获取要在创建窗口的线程上，进一步拆分离屏工作、呈现与资源同步需要 Mojive 自己实现。[bgfx 线程模型](https://bkaradzic.github.io/bgfx/internals.html)、[SDL 命令缓冲](https://wiki.libsdl.org/SDL3/SDL_AcquireGPUCommandBuffer)、[SDL 交换链线程约束](https://wiki.libsdl.org/SDL3/SDL_WaitAndAcquireGPUSwapchainTexture)
3. **SDL 拾取优势不足以抵消本阶段的集成成本。** 它的显式 fence 很合适做读回；若以后业务以大批量离屏导出或更严格的拾取延迟为主，可以依据同一套测试重新选型。当前选择并不否认其优势。
4. **避免把 shader、窗口和物理库混进公共接口。** 当前没有修改上游 bgfx、SDL 或 ImGui 源文件。两个后端的交换已经证明接口可替换；选择 bgfx 不需要把 bgfx handle 或 view ID 带入 Session、场景源或 Python API。

两者都能承载传统 raster rendering；SDL GPU 不是旧的 SDL 2D renderer。本次没有发现当前 Mojive 功能必须依赖、而另一候选完全不具备的通用 raster 能力。选择依据是实现成本、实测调度与维护风险，不是 API 名称。[SDL GPU 范围](https://wiki.libsdl.org/SDL3/CategoryGPU)

## 生产渲染的迁移边界

本次数值与合成测试验证了依赖的基础操作，**没有完成生产阴影、反射、材质和完整 UI 的逐像素迁移**。这些属于选型后实现的验收条件，不能将基础 pass 通过写成整套画质通过。

| 现有功能 | C++ 实现方向 |
|---|---|
| 不透明/透明材质、2D/cube 纹理、雾与天空 | 保留 Mojive 材质含义，按材质分桶；透明 pass 明确排序，固定线性色彩与输出编码 |
| 阴影、级联、反射 | 由 Mojive render pass 计划管理相机、目标与采样依赖；在 bgfx 适配层映射 view 顺序 |
| 描边、gizmo、debug 和标签 | 保留选择 identity 与视觉语义，使用独立 overlay pass；不让 UI 直接操作 backend handle |
| wireframe、宽线与特殊几何 | 使用可在 Metal/D3D12/Vulkan 复用的三角形或 shader 展开，不照搬仅某个 API 支持的 geometry stage |
| skin/flex/动态网格 | 稳定 topology 时复用 GPU buffer；topology 变化通过结构版本更新，旧票据按版本作废 |
| 多相机输出 | 固定缓冲容量与显式背压；交互可以保留最新请求，离线导出必须完整消费 |

SDL 的四个颜色附件上限能容纳当前实验的数据 pass；原生整数附件格式能力与实际整数 shader 输出是两回事，本轮实际使用了经过正确性验证的无损 RGBA8 identity 编码。bgfx 的五个便携 instance vec4 不应变成公共数据结构限制；生产材质数据采用批次常量或独立 GPU 数据表，不强行把全部 Python instance 布局塞入它。

## 内存与生命周期风险

窗口功能与取消逻辑通过，不等于已经证明没有 GPU/平台资源滞留。反复开关窗口时两个实现的进程物理 footprint 都上涨；完整销毁 device 后明显回落。纯 SDL 程序去除 GLFW、Mojive 和 ImGui 后也能观察到增长。缓存、平台保留和实际泄漏不能仅靠 footprint 区分，本轮不将它伪装成“内存全部通过”。原始序列保存在本地报告。

已将正常缩放改为保留 surface 的原地更新，避免不必要的重建。生产阶段复用渲染目标、交换链、上传与读回缓冲，并用更长时间的稳定工作负载及平台 profiler 检查平台资源；不要把短时窗口 churn 的峰值当成常驻编辑器内存。

## C++ 开发顺序

1. **核心与运行时。** 标准 C++20 的场景、命令、选择 identity、快照和任务完成类型；物理 adapter 独占可变物理状态。资源线程池负责加载与解码，完成后交给 owner 应用。Session 统一拥有选择和编辑命令；渲染快照可以取最新值，编辑命令则按顺序确认，不能因丢旧帧而丢命令。异步结果携带结构版本、请求 identity 和取消状态。
2. **渲染和 Python 薄接口。** 完成 bgfx 生产 pass，与现有 `SceneSource` / `SceneFrame` 语义对齐。nanobind 承担批量提交、拥有明确生命周期的数组和异步结果；MuJoCo 官方 pybind11 绑定不变，不跨库混用其 C++ 包装对象。
3. **原生 ImGui 编辑器。** UI 只消费快照和发送命令，保持 JetBrains Mono + Noto Sans CJK；公共组件建立统一间距、对齐和 4.8 px 常规圆角，自绘组件再处理曲率连续边缘。字体和布局由组件层负责，RHI 不负责解决这些问题。
4. **CLI 与可连接 session。** GUI 和 CLI 共用核心库；有远程、多客户端或后台持续模拟需求时启动独立 session runtime。第一阶段不强制所有本地窗口都走 IPC，也不把“多个进程共享程序代码页”混同于共享场景、物理堆和 GPU 资源。

macOS 选择 Metal，Windows 选择 D3D12，Linux 选择 Vulkan；保留适配器替换能力。平台层当前保持 GLFW，不因为 SDL GPU 实验引入第二套生产窗口事件系统。底层资源与 shader 编译器版本按锁文件一起升级。三平台的构建、真实 GPU 与发布包验证必须完成后，才能宣称原生版支持三平台。

复现命令见[原生验证指南](../guides/native-probe.md)。历史比较和初始方案见[原生迁移提案](native-cpp-bgfx.zh.md)，其中未完成事项以本文最新状态为准。
