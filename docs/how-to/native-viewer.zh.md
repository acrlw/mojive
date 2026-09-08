# 运行 C++ / bgfx Viewer

原生后端通过现有 Python Viewer、`Renderer` 和 `SceneRenderer` 使用。bgfx 直接调用系统图形 API：macOS 使用 Metal，Windows 使用 Direct3D 12，Linux 使用 Vulkan。它不经过 wgpu，也不需要 Dawn。OpenGL 和 wgpu 后端继续保留，用于兼容现有脚本、回归对照和独立选择；开发 bgfx 功能时不必再实现一层 wgpu。

## 启动

在开发 worktree 中运行：

```bash
make native-viewer
make native-editor
make native-viewer SCENE=/path/to/model.xml
make native-viewer SCENE=joint_gizmo ARGS="--play --no-vsync"
```

这些目标编译 C++ 扩展和 shader，然后启动原有 Python UI。首次编译较慢，后续使用增量构建。`make native-python-build` 只构建扩展；独立 C++ probe、gallery 和 runtime 目标仍然保留。

新 checkout 需要 CMake、Ninja、C++20 编译器，以及对应平台的图形开发环境。先初始化仓库锁定的依赖，再安装 Python 环境和现有 ImGui 定制：

```bash
git submodule update --init --depth 1
uv sync --frozen --extra dev --extra mujoco
make setup-imgui
make native-viewer
```

不需要初始化上游嵌套 submodule，不使用 `--remote`。本轮没有添加或启动 workflow CI。

## Python 接口

Make 自动设置开发扩展路径；直接运行自己的脚本时设置：

```bash
export MOJIVE_NATIVE_BUILD="$PWD/output/cpp-build"
```

原有调用方式保持不变：

```python
import mujoco
import mojive

model = mujoco.MjModel.from_xml_path("assets/joint_gizmo.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with mojive.Renderer(model, width=640, height=480, renderer="bgfx") as renderer:
    renderer.update_scene(data)
    rgb = renderer.render()
    future = renderer.render_async()
    next_rgb = future.result()
```

`SceneRenderer(..., renderer="bgfx")`、`build(..., renderer="bgfx")` 和 `MOJIVE_RENDERER=bgfx` 同样可用。原来的 Make 场景目标使用 `BACKEND=bgfx`，同时保留上述扩展路径。

RGB 为左上原点 `uint8`，metric depth 为相机前向的 `float32` 世界单位深度，object ID 为 `uint32`，分割为 `int32` 二元组。颜色 MSAA 请求向下取 1×、2× 或 4×（0 表示不抗锯齿），更大的模型请求不会导致加载失败。矩阵继续按行主序解释，平移位于 `matrix[:3, 3]`；用户不需要转置。已有 `out` 缓冲和非连续目标数组继续可用。

异步读取返回 `concurrent.futures.Future`，可交给 `asyncio.wrap_future()`。它保留提交时的帧，普通姿态／相机更新不会取消它；缩放、替换场景结构或释放目标会取消尚未完成的读取并报告异常。返回数组不依赖 GPU 对象寿命。Future 完成前不能修改其 `out`。

## 已实现的渲染能力

- 两种颜色模型、ambient、headlight、方向／点／聚光／面积光、image light、雾和 horizon haze。
- 实例材质、二维／立方体纹理、线性光 mipmap、透明排序、透明物体的可选身份输出。
- 方向光级联阴影、局部光阴影、三种阴影质量；最多四组平面反射和 box 顶面反射。
- 天空盒、选择轮廓／xray、3D gizmo、九条 debug 绘制路径、文字及物理诊断可视化。
- 所有 `RenderFlag`、五种 debug view、动态 mesh／skin／flex、tendon 和 actuator 配色。
- 颜色／身份输出请求裁剪、阴影／反射缓存、复用实例上传缓冲、独立场景和多个窗口。

选择某个后端不代表 UI 改成 C++：当前面板、输入、Session 和业务扩展继续使用 Python。C++ 提供渲染、设备与资源寿命、提交和异步读回基础设施。ImGui 源码已在仓库管理，后续可以独立定制其绘制；本轮没有重做 UI 或引入曲率连续圆角。

## 线程与所有权

- 窗口事件、ImGui 和 Session 状态在原有 Python UI 线程执行。
- 进程级 C++ runtime 独占 GPU API 调用，bgfx 渲染线程执行后端工作。等待原生工作和读回时释放 GIL。
- 多个 Viewer、Renderer 和相机共享设备，场景及输出相互独立。关闭一个实例不会重建其他实例的资源。
- UI 直接采样 GPU 场景纹理，普通窗口显示不把整帧读回 CPU。
- 原生日志保存在有界历史中，由 Python 调用线程按游标转发到既有 Loguru／Output；worker 不回调 Python。
- Mojive 管理的 MuJoCo 仿真使用已有并发驱动。调用者自己拥有的 `MjData` 仍需保证写入与 `update_scene()` 读取互斥。

VSync 直接控制 GPU 呈现同步，不使用 sleep 模拟，也没有固定的 60 FPS 上限；节奏取决于窗口所在显示器当前启用的刷新模式。GPU 提交队列限制为一个帧，呈现表面保留两个 drawable，减少过时画面排队。bgfx 的 VSync 是共享设备策略：同一进程中只要一个窗口开启，就保持所有共享窗口同步；关闭该窗口或关闭其 VSync 请求后重新计算。`--no-vsync` 用于显式无同步运行。

Python 面板和用户回调仍受 GIL 约束。当前没有引入 daemon、远程 session runtime 或跨进程崩溃隔离。macOS 的原生等待会服务主线程系统队列，避免 AppKit 窗口创建与渲染线程相互等待。

## 验证和安装包

```bash
make native-parity
make native-motion-parity
make native-window-benchmark
make native-ui-parity
make native-load-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
make native-corpus-parity MENAGERIE_ROOT=/path/to/mujoco_menagerie
make native-model-parity HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make native-viewer-test HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make native-features-test
make native-spirv
make native-wheel-test
```

`native-parity` 比较颜色、深度、object ID、分割及功能开启／关闭的可见贡献；`native-model-parity` 还检查真实刚体、skin、flex、变形恢复和 RGBA。对照图从左到右为 OpenGL、wgpu、bgfx。截图、JSON 和性能数据位于 `output/`。详细结论见[本轮验收记录](../plans/native-renderer-parity.zh.md)。

`native-motion-parity` 保存连续近景平移、环绕、缩放的逐帧 PNG、并排动画和指标；`native-corpus-parity` 通过公开 MuJoCo Renderer 逐个加载本地模型，默认在单采样下严格对照八个视角的 RGB 和分割，MSAA 图像另由 motion／Viewer 验收覆盖；生成每模型最差视角、总览图和失败清单。缺少依赖或非独立 XML 片段会单独报告，不能计作通过。

`native-window-benchmark` 在真实可见窗口中分别开启／关闭 VSync，测量连续平移、环绕和缩放的 CPU 提交帧时间，并检查相机变更在同一帧送到渲染器。这不是光子延迟测量，CPU 提交 FPS 也不等于显示器实际呈现 FPS。测量时避免同时运行其他 GPU 验收。

`native-ui-parity` 检查轴球与文字在点击转场中的连续相对位置，以及中文／英文、圆角轮廓和线条的抗锯齿。`native-load-benchmark` 在独立进程中比较十种 Menagerie 机器人的首次加载与同进程重复加载，分别记录解析、资源准备、首帧提交和首帧可读取的时间；不会把解析时间当作整个等待时间，也不会把 GPU 读回时间称为实际屏幕显示时间。系统文件缓存未清除。

原生纹理使用仓库已有的 stb resize，在释放 GIL 后生成线性光 mipmap；较大的纹理集合使用有界线程池。完成的像素数据以不可变共享存储交给上传任务，避免额外复制，并保证调用方释放后 GPU 仍能读取。

`native-wheel-test` 构建平台专用 wheel，安装到独立目录，移除开发环境变量后实际渲染。安装这个 wheel 后无需设置 `MOJIVE_NATIVE_BUILD`。普通 `uv build` 仍生成原有纯 Python 包；只有显式设置 `MOJIVE_NATIVE_WHEEL_BUILD` 才附带扩展、shader 和原生依赖许可证。当前 wheel 匹配构建机器的 Python ABI、架构和操作系统版本，不是跨平台通用包。

本机实测为 macOS／Metal，包括 Retina 2× framebuffer 和 150% 中文 UI。Vulkan 的全部 shader 已编译为 SPIR-V；Linux／Vulkan 和 Windows／D3D12 的实际设备运行仍需在对应机器验收。Linux 窗口当前使用 X11（Wayland 桌面可经 XWayland），尚未提供原生 Wayland 窗口路径。

## 鼠标映射和后端能力

Settings → Interaction 的 Mouse gestures 和现有键盘映射共用一份输入配置。可以选择
Mojive、Blender、Unity、Unreal、MuJoCo 的视角操作预设，也可以为每项编辑组合，例如
`alt+left; middle`、`left+right`、`shift+right`、`left:double`。分号分隔替代操作；回车应用；
清空表示解绑。视口、时间轴、面板、滑块和工具栏具有各自的操作上下文，同一上下文冲突会拒绝保存。
预设只覆盖导航习惯，不表示完整复制对应软件的所有工具、第一人称导航或快捷键。

```python
viewer.configure_navigation_preset("Blender")
viewer.configure_pointer_binding("camera.pan", ("left+right",))
viewer.configure_pointer_binding("timeline.pan", ("middle",), persist=True)
```

拖动开始后保留取得操作权的按钮组合，释放其中一个按钮结束组合拖动，并等剩余按钮全部释放后
再接受新的场景操作。修改映射也会更新状态提示。普通 ImGui 控件激活、文本编辑和系统菜单快捷键
仍遵循控件及平台约定；新语义鼠标操作应通过 `PointerAction` 接入，不要在处理器里新增按钮常量。

模型导入入口由当前物理适配器的格式声明决定，不再统一假设 MJCF / URDF。
未实现扰动的后端不会启用对应鼠标操作；公共命令、扩展能力和 RPC 版本要求见
[适配器契约](custom-adapter.md#capability-and-version-contracts)。

## G1 多 world 演示与压力测试

```bash
make g1-worlds MENAGERIE_ROOT=/path/to/mujoco_menagerie G1_WORLDS=1024
make g1-worlds-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-worlds-transport MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-worlds-monitor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
```

前两个目标编译原生扩展。演示默认使用 bgfx；`ARGS="--renderer opengl"` 可切换交互演示。
原始精度是默认值。基准先比较 1、4、16 个机器人近景的 RGB、深度和身份，再计时
1024、2048、4096 个 world。动作来自 Unitree 官方 `unitree_rl_mjlab` 的
`dance1_subject2.csv`（Apache-2.0，提交 `1425b15f73bd4095f0df53709d7c389c3eb9e790`），
下载后验证 SHA-256；不会执行下载的脚本。模型使用本地 Menagerie 的 `unitree_g1/scene.xml`。

这是独立姿态回放，非 4096 个物理仿真，也不是训练性能。每个 world 有独立随机起始相位，
固定种子，默认间距 7 m。网格资源共享，绘制按实例分批。1024 个原始机器人已经包含约
4.03 亿个三角形／场景 pass，不能期待仅改用 C++ 就能以 120 FPS 绘制。

可显式评估 meshoptimizer LOD：

```bash
make g1-worlds-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie \
  ARGS="--mesh-ratio .01 --mesh-error .05 --output output/g1-worlds-lod"
```

误差上限包含法线和 UV 属性的约束，实际减面比例可能高于目标。它不等价于像素误差，
更不能保证近景看不出差异。LOD 只在显式请求时生成，两种后端使用同一份结果；原始网格不会覆盖。
报告区分姿态更新、renderer 更新、完整 GPU 输出及读回、首帧和内存。显示帧率、物理吞吐、
离屏输出吞吐和传输延迟分别报告，不能互相替代。

`g1-worlds-transport` 只测传输；`g1-worlds-monitor-benchmark` 在独立发布进程之外，
还执行接收、renderer 更新、GPU 绘制和完整图像读回。两者都使用 localhost TCP、120 Hz
目标发布速率和 30 Hz 目标接收速率。报告分别记录取到快照时的年龄和 GPU 图像完成时的年龄；
后者包含渲染等待，仍不是屏幕扫描延迟。`ARGS="--renderer opengl"` 可运行 OpenGL 对照。
所有进程由本次测试创建并在结束时关闭，不会附着到已有 session。

原生阴影 pass 对每级光源视锥做保守包围盒剔除，保留相交物体；移动、负缩放和动态网格更新
会刷新包围盒。它不改变原始网格精度，颜色和反射 pass 不使用这项剔除。
