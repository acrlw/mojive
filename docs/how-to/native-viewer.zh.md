# 运行 C++ / bgfx Viewer

本分支已把原生渲染接入现有 Python Viewer。可以直接启动、加载模型、操作原有面板和测试 Python API；默认 OpenGL 后端仍保留。

这是**可运行的原生后端预览**，还不是与原后端画质和功能完全等价的替代品。基础材质、二维纹理、实例化、选择 ID、深度和分割输出已接入；完整灯光、阴影、反射、天空盒、立方体纹理、透明物体排序、原生调试绘制和原生 3D gizmo 仍需后续迁移。未支持的可选开关通过 capabilities / 返回值明确报告。

## 启动

当前开发 worktree 已配置独立的 `.venv`，进入目录后运行：

```bash
make native-viewer
make native-editor
```

第一个命令编译 C++ 扩展和 shader，然后使用现有 Viewer 打开默认场景。第二个打开空白场景编辑器。首次编译需要较长时间，之后是增量构建。

指定模型或播放状态，继续使用原来的参数：

```bash
make native-viewer SCENE=/path/to/model.xml
make native-viewer SCENE=joint_gizmo ARGS="--play --no-vsync"
make native-editor ARGS="joint_gizmo --no-vsync"
```

`make native-python-build` 只构建扩展；原来的 `make native-build`、`native-probe` 和 `native-gallery` 仍然用于独立 C++ 验证。

在新 checkout 中先运行 `uv sync --frozen --extra dev --extra mujoco`，然后运行 `make setup-imgui` 取得项目原有的 UI 绘制与焦点修复；并安装 CMake、Ninja 和支持 C++20 的编译器。依赖源码已固定在仓库中；submodule checkout 使用 `git submodule update --init --recursive`。本次没有启动 workflow CI。

## 现有 Python 用法

Make 会自动设置开发扩展路径。直接运行自己的脚本时设置一次：

```bash
export MOJIVE_NATIVE_BUILD="$PWD/output/cpp-build"
```

随后仍使用原 API：

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

`SceneRenderer(..., renderer="bgfx")` 和 `build(..., renderer="bgfx")` 同样可用。也可以设置 `MOJIVE_RENDERER=bgfx`，继续调用原有入口。`make viewer` 使用自己的 `BACKEND` 变量；使用 `BACKEND=bgfx` 时同时保留上述 `MOJIVE_NATIVE_BUILD`。

RGB 是左上原点的 `uint8` 数组，深度是 `float32` 世界距离，object ID 是 `uint32`，分割是 `int32` 二元组。矩阵与原来的行主序含义一致，用户不需要转置。已有 `out` 缓冲和带 stride 的数组继续可用。

异步读取返回 `concurrent.futures.Future`，可交给 `asyncio.wrap_future()`。读取提交后会保留对应帧；目标缩放、场景结构替换或释放会使未完成读取取消并报告异常。返回的数组不依赖 GPU 对象寿命。不能在 Future 完成前修改其 `out`。

当前原生预览未实现阴影。默认 `BALANCED` 预设不会阻止无阴影后端初始化；显式设置阴影预设以及构造时请求非默认预设仍报告不支持。

## 线程与所有权

- 窗口事件、ImGui 和 Session 状态仍在原来的 Python UI 线程执行。
- 一份进程级 C++ runtime 独占 GPU API 调用，bgfx 的渲染线程执行后端工作。等待原生工作和图像回读时释放 GIL。
- 多个 Viewer、Renderer 和相机目标共享设备，但场景、网格、材质和输出相互独立。关闭一个实例不重建另一个的资源。
- UI 直接采样 GPU 场景纹理；窗口先在 GPU 上合成，再通过一个全屏四边形呈现。普通显示不会把整帧读回 CPU。
- 原生记录保存在有界日志历史中，由 Python 调用线程按游标转发到现有 Loguru / Output；C++ worker 不回调 Python，也不更改宿主的 Loguru 配置。
- Mojive 管理的 MuJoCo 仿真继续使用已有并发驱动。传入用户自己的 `MjData` 时保留现有所有权约定，不能一边写它一边要求 `update_scene()` 读取它。

这消除了 GPU 工作依赖 Python GIL 的部分限制，但不意味着 Python 面板和用户回调都能并行执行。当前也没有引入 daemon、跨进程 session 或崩溃隔离。

macOS 的 Metal 窗口创建需要主线程处理系统任务。原生等待过程会服务主线程系统队列，避免 UI 等待渲染、渲染又等待 AppKit 的启动死锁。

## 验证与边界

```bash
make native-viewer-test HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make cpp-python-gpu
make check
```

完整 Viewer 验证会保存 `output/native-viewer/viewer.png`、`selection.png`、`humanoids100.png`、`scaled-cjk.png` 和 `acceptance.json`，覆盖独立场景、异步读取、多窗口缩放/关闭/重开，100 humanoid 的物理帧推进和 150% 中文 UI 缩放。本次显示器的 framebuffer 倍率为 1，真实 Retina 显示器仍待测试。

目前只在本机 macOS / Metal 上实测。Windows、Linux、Wayland 原生窗口支持和可分发的原生 wheel 不算已验收；当前 Linux 窗口路径使用 X11。不能把当前预览与生产 OpenGL 的 FPS 直接比较，因为完整画质尚未对齐。
