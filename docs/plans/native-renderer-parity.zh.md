# 原生渲染完整性与 parity 验收

本轮在 `codex/cpp-foundation` 修复并验证 Python Viewer 使用的 C++ / bgfx 后端。macOS Metal 的下列验收已通过，可以用 `make native-viewer` 启动；这不代表已经完成 Linux、Windows 或 120 Hz 显示模式的设备验收。本轮没有启动 workflow CI，也没有合并到 main。

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

| 验收 | 结果与证据目录 |
|---|---|
| `make check` | Fast、integration、分层、文档和示例检查；日志 `output/native-delivery-check.log` |
| `make gpu`、`make gpu-wgpu` | OpenGL／wgpu 回归通过；日志 `output/native-final-opengl.log`、`output/native-final-wgpu.log` |
| Physics、MuJoCo audit、adapter conformance | 413 项通过、2 项跳过，audit／conformance 通过；`output/native-physics-audit.log` |
| C++ CPU contracts | 3 项 CTest 通过；`output/native-core.log` |
| 私有原生 render、feature、完整 Viewer | 31 项通过，包括真实窗口、Retina／中文、100 humanoid、resize、VSync 和异步读回；`output/native-final-regression.log` |
| `make native-parity` | 96 组功能对照，含开关必须产生实际可见贡献；`output/native-parity/` |
| `make native-model-parity` | 16 组刚体、skin、flex、100 humanoid 的动态／恢复、线框／着色对照；`output/native-model-parity/` |
| `make native-motion-parity` | 204 对连续近景平移、环绕和缩放图；两种颜色模型、0／4 samples，全部通过；`output/native-motion-parity/` |
| `make native-corpus-parity` | 251 个有可绘制内容的模型／场景，各 8 个视角，颜色和经深度分类的分割验收通过；`output/native-corpus-single/` |
| Vulkan shader | 40 个 shader 编译为 SPIR-V；`output/native-spirv.log` |
| `make native-wheel-test` | 构建平台 wheel，隔离安装后移除开发路径变量，以公开 Renderer 实际生成 RGB；`output/native-wheel-final.log` |

本机本地语料共发现 295 个模型文档：251 个有可绘制内容、18 个空文档（包括 `assets/empty.xml` 和只含定义的片段）、26 个不能独立编译的 include 片段。18 个空文档通过加载、组合和背景检查，但不计入机器人渲染数量；26 个片段单列为 skipped，不计作通过。这修正了先前把 269 个可加载文档统称为完整模型的口径。

模型语料默认单采样，单独检查物体内部颜色，保留整幅 RGB 差异和原始身份差异；MSAA 边缘由 motion、feature 和真实 Viewer 检查覆盖。11 个视角的原始分割差异超过 0.1%，来自重叠／等深手部几何和远裁剪边界：两个后端的投影深度差在四个 float32 深度缓冲 ULP 内。报告保留 `segmentation_disagreement`，分别记录 `depth_tied_identity`、`far_clip_identity` 和未解释差异，不将其描述为逐像素身份完全一致。其他视角不需要这一分类才能通过原始分割门槛。

连续相机对照的最大整幅颜色平均差异为 0.0213／255，P99 为 0，最大逐帧误差变化为 0.0404／255。已经查看近景方块、球面、阴影／反射、模型总览和真实 Viewer 的输出图。动态恢复测试同时检查相机／姿态恢复后的输出，防止缓存留下旧画面。

代表性文件：

- `output/native-motion-parity/mujoco-classic-msaa4/pan-08-comparison.png`：左 OpenGL、右 bgfx，检查方块侧面。
- `output/native-motion-parity/mujoco-classic-msaa4/motion.webp`：连续运动对照。
- `output/native-viewer/projection-bgfx-820-880.png`：窄窗口的实际投影。
- `output/native-model-parity/100_humanoids-shaded-1-comparison.png`：三后端大量刚体对照。
- `output/native-corpus-single/contact-00.png` 至 `contact-13.png`：模型总览，每对左 OpenGL、右 bgfx。

## 刷新率和交互延迟

没有固定 60 FPS 上限，也没有根据显示器“最高规格”强行提交的定时器。开启 VSync 时由 GPU／系统的当前显示模式安排呈现；切换显示模式后不依赖缓存的刷新率。实际检查 CAMetalLayer 的 `displaySyncEnabled` 为开／关／开，并验证关闭同进程的另一个窗口不会丢失剩余窗口的同步策略。

本次系统报告 Apple M5、AG25Q380，4096×2304 像素、2048×1152 逻辑分辨率，当前为 **60 Hz**。因此不能宣称已验证 120 Hz 实际呈现。没有修改用户的系统显示设置。OpenGL 的 CPU 提交计数可能高于显示刷新率，不能据此认定显示器运行在 120 Hz。

`make native-window-benchmark` 在可见窗口中连续改变相机，验证变更在同一帧交给 renderer，分别测量 VSync 开／关的三轮帧间隔；结果在 `output/native-window-benchmark/report.json`。这是 CPU 提交节奏，不是输入到光子的硬件延迟测量。手动平移和环绕使用直接相机变更；聚焦动画与手动操作分开。

可见窗口固定为 1200×800 逻辑像素，viewport 为 1306×1036 设备像素，三轮结果如下：

| 后端／模式 | CPU 提交 FPS 范围 | 单帧中位数 | 单帧 P95 |
|---|---:|---:|---:|
| bgfx，VSync 开 | 60.2–60.3 | 16.61–16.62 ms | 16.86–17.04 ms |
| bgfx，VSync 关 | 165–185 | 4.69–4.99 ms | 12.14–16.64 ms |
| OpenGL，VSync 关 | 301–327 | 2.47–2.68 ms | 6.27–6.47 ms |

这个轻量场景中原生后端的 CPU 提交仍比 OpenGL 慢，尤其存在呈现相关的长帧，不能宣称所有场景都因 C++ 更快。解除固定限帧、接入真正的呈现同步和减少排队，分别解决不同问题；没有硬件延迟测量就不能把提交时长当成最终手感提升的量化结论。

## 性能与平台边界

性能必须在相同投影、画质、尺寸和仿真轨迹下比较。早期 Viewer 投影错误时的跨后端数据不作为最终速度结论。100 humanoid 通过生产 Session 分别运行 serial 和 threaded，每种三轮，结果在 `output/native-performance/current-bgfx/` 和 `current-opengl/`；同时保存与串行物理重放一致的检查结果、截图和时间线。

100 humanoid 使用同一模型、1044×740 viewport、VSync 关闭、目标节奏 120 FPS，每轮 3 秒，三轮中位数：

| 后端／物理模式 | 渲染提交 FPS | 单帧工作 P95 | 物理 step/s |
|---|---:|---:|---:|
| OpenGL，串行 | 63.0 | 20.80 ms | 198.2 |
| OpenGL，并行 | 117.1 | 10.47 ms | 199.6 |
| bgfx，串行 | 99.3 | 14.07 ms | 198.4 |
| bgfx，并行 | 119.7 | 6.57 ms | 199.1 |

并行时 bgfx 的单帧工作 P95 比 OpenGL 低约 37%，两者帧率都接近测试的 120 FPS 节奏上限，不能据此推算无上限吞吐提升。这里的 120 是 benchmark 的目标调度频率，不是 Viewer 的产品限帧。物理状态全部通过逐步串行重放验证。

bgfx 直接使用 Metal／D3D12／Vulkan，不经过 wgpu 或 Dawn。Linux 构建路径选择 Vulkan，当前窗口路径为 X11／XWayland；尚未提供原生 Wayland 窗口接入。Windows D3D12、Linux Vulkan 和 120 Hz 显示模式仍需要对应环境实测；本轮 SPIR-V 编译不能替代设备验收。wheel 匹配本机 CPython ABI、arm64 和 macOS deployment target，不是跨平台通用包。
