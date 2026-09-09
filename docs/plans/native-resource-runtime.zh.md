# 原生资源与 runtime 验收

本阶段继续在 `codex/cpp-foundation` 上开发，基于 `52360da`，范围为 macOS／Metal 和 Linux／Vulkan。Windows 暂缓；没有启动 CI，也没有修改 main。

## 已实现

- 取消异步 Future 后仍回收原生读回 ticket，输出缓冲保持不变；共享设备最多八个读回，满队列立即返回背压错误。
- 等待中的原生异步读回让出 owner 给其他已接受任务；关闭 runtime 仍等待资源清理。
- 输出目标按 GPU 帧租用 view ID，解除十二个长期目标的限制；四十目标交错读取、缩放和释放通过。
- 不可变网格与纹理按身份／内容复用 CPU 和 GPU 存储。等价重编译不重复上传；独立场景变形使用独立 GPU 几何。
- 模型加载、关键帧和源码编辑共用现有工作线程队列，后台准备网格、摘要与 mipmap；GPU 提交继续由 owner 执行。
- shader 编译改为后台任务，成功后整组发布；编译失败保留旧程序，关闭时终止并等待自有编译进程组。
- Metal 离屏 viewport／scissor／winding 使用统一方向，修复精确三角形边界和 4×／8× MSAA 覆盖差异。Shared 读回缓冲不再执行仅适用于 Managed 的同步操作。
- 更新 macOS 输入验收：测试通过实际窗口输入适配器注入事件，并遵循 ImGui 的平台修饰键语义。

## 验收与测量

本轮记录来自 Apple M5、macOS 26.6.2、Python 3.11.15、MuJoCo 3.11.0 和定制 ImGui 1.92.900+mojive.1。原始结果位于 `output/native-refinement/continuation-*`；以下是新增 UI 调整之前完成的原生验收。

- CPU／集成：`make check` 1708 + 146 项；C++ CTest 6 项；绑定 CPU 8 项、GPU 21 项。
- OpenGL GPU 477 项、wgpu GPU 360 项、bgfx GPU 420 项；独立原生功能／shader／Viewer 回归 54 项。各 GPU 组隔离进程执行。
- 96 个功能对比、16 个模型／形变对比和 204 帧连续平移／环绕／缩放均通过；bgfx 的最大颜色平均绝对误差分别为 0.003715、0.009242、0.007263（8-bit 通道，满量程 255），P99 差异为零。已目视检查近距离方块、反射、100 humanoid 和 CJK/轴标图像。
- Metal 精确覆盖及 MSAA 0／2／4／8 倍测试通过，Metal API 验证通过；45 个 Vulkan SPIR-V shader 编译通过。macOS wheel 独立安装后可以渲染。这些不等于 Linux 设备验收。

### 渲染与加载

公开 Renderer 同步取图矩阵覆盖 7 个场景、640／1280／1920 三种宽度和 RGB／depth／segmentation，共 63 对；每项预热 10 帧、测量 80 帧。bgfx / OpenGL 的中位耗时比为 RGB **0.315**、depth **0.421**、segmentation **0.459**，该矩阵中 63 对均更快。它包括静态资源复用收益，不能直接当成交互延迟或所有动态场景的倍率。

10 个 Menagerie 模型各重复 3 次；统计从请求到首张 GPU 可读图像，包含资源准备。文件缓存未清空。代表结果：

| 场景 | OpenGL 首图 / 最长 UI 帧 | bgfx 首图 / 最长 UI 帧 |
|---|---:|---:|
| ANYmal C | 3049.9 / 2633.6 ms | 393.3 / 47.1 ms |
| Unitree G1 | 177.1 / 46.6 ms | 132.4 / 16.4 ms |
| Franka Panda | 382.3 / 58.4 ms | 307.8 / 21.0 ms |

MS-Human-700 独立编辑场景中，首次选择 OpenGL / bgfx 为 9.61 / 9.55 ms，重复选择 6.89 / 5.08 ms，平移 6.46 / 4.78 ms；选择、关键帧和等价源编辑未重复上传 GPU 网格／纹理。没有复现原来的约 3 秒选择停顿。部分轻模型的再次加载仍慢于 OpenGL，不能宣称每项操作均有提升。

### 尚未消除的编辑长帧和测量限制

后台捕捉关键帧、删除模型、修改源文件的 bgfx 最长 UI 帧仍为约 99／61／50 ms（同场景 OpenGL 94／58／54 ms）。采样发现部分长帧与 MuJoCo `MjSpec.copy()` 重叠；当前 Python 绑定没有在该函数释放 GIL。资源缓存解决了重复上传，却没有消除拓扑编辑的所有停顿。后续应以进程隔离编辑或经过验证的上游绑定改进解决，不能对现有 Python 对象用未经验证的裸指针绕过 GIL。

100 humanoid（1600 个运动刚体，2700 DOF）采用同模型、1392×1036 视口、200 physics steps/s 目标、串行和并发交错运行三轮。并发渲染 FPS：OpenGL **107.2 / 89.9 / 43.8**，bgfx **118.6 / 69.3 / 53.0**；对应物理 steps/s 为 **199.9 / 199.7 / 90.4** 和 **199.4 / 122.4 / 96.3**。轨迹验证通过，但宿主负载波动很大，第二轮 bgfx 较慢，不能据此宣布全场景“不比 Python 慢”。

同分辨率真实窗口无 VSync 测量的提交速率中位数为 bgfx 309 FPS、OpenGL 238 FPS，但部分运行失去窗口焦点，因此只作吞吐参考。环境报告的显示刷新率为 60 Hz；未完成 120 Hz 实际显示／输入到扫描输出的验收。

## UI completion and measurements

The current UI follows the final interaction requirements: soft-white capsule outlines; explicit
playback and tool groups; independent temporary physics snapshots; compact Output expansion;
colored active hierarchy filters; and consistent numeric/unit controls across Camera, Inspector,
Joints, and Control. Wrapped rows fill the available width. Earlier requests for continuous field
width through a layout breakpoint were superseded by the full-width wrapping requirement.

Name editing uses double-click, Enter/blur to commit, and Escape to cancel. Single-click selection
leaves tools unchanged; double-clicking a joint or actuator focuses and enables its applicable
gizmo. Camera synchronization and bookmark paste reuse the scene-camera command route.
Diagnostic and icon geometry is cached. The final editor checklist and captures are in
`output/ui-refinement/checklist.md` and `output/ui-refinement/final-layout-bgfx/`.

Earlier native UI parity checks found moving axis-label centers within 0.000008 logical pixels
and P99 color difference zero at normal and 150% scale (`output/ui-refinement/ui-parity/`).
The earlier MS-Human-700 editor comparison used matched 653×518 viewports: first selection was
15.652/8.589 ms and panning 7.032/4.781 ms for OpenGL/bgfx. Native selection and equivalent
recompiles uploaded no repeated meshes/textures. Background source/keyframe edits still produced
approximately 30–48 ms long frames; these measurements do not establish universal speedups.

## 仍需完成的范围

- 本轮在 macOS 运行；Linux 新版资源和 runtime 行为需对应设备复验。已有 Linux X11／原生 Wayland 路径继续保留。
- 大资源首次 GPU 上传仍是一项 owner 任务；尚无按帧预算分段上传。后台 CPU 准备不等于零 UI 停顿。
- 没有自动屏幕误差 LOD；4096 个原始 G1 的性能边界仍见前阶段报告，显式减面不能冒充原始精度。
- vendored ImGui 与 Python imgui-bundle 仍分别构建，完整控件的曲率连续圆角设计未统一接入。
- 没有新增常驻原生 daemon 或跨进程崩溃恢复；已有 Python RPC／serve／attach 能力不受影响。

## Additional corpus and UI acceptance

The final 4x MSAA corpus covers 295 local assets/Menagerie XML documents: both backends load
and render 269 successfully and skip 26 include fragments. These include 251 renderable models
and 18 empty documents. Eight views are compared per available document. Two strict per-object
color checks still fail: `aloha/scene.xml` view 07 and `low_cost_robot_arm/scene.xml` view 01.
This corpus is **not a parity pass**; raw results are in
`output/native-refinement/continuation-corpus/report.json`.

Failing-object interior errors are 1.720/3.282 on the 0-255 scale; whole-image means are
0.040/0.023. The matching 16 views pass without MSAA. Isolated albedo colors match, normal
outputs differ, and disabling shadows does not eliminate the differences. Actual 4x sample
positions match across OpenGL/Metal; an independent thin-triangle normal fixture differs by
at most one color level. A centroid-normal shader experiment increased failures and was
rejected. Production shaders and acceptance thresholds are unchanged; depth/mesh seam
causes remain under investigation.

A float-depth OpenGL probe and a native precomposed view/projection probe produced no changed
pixels in their respective reference images. Neither explains the remaining MSAA discrepancies.
No production tolerance, object mask, or shader was weakened to report a pass.

The UI palette is Info `#8AB7C0`, Warning `#C9A15C`, Error `#D06744`. Warning uses the
continuous-curvature 0.618 contour. Inner marks and outer frames have distinct widths, and
Retina antialiasing uses a one-physical-pixel fringe. Scalars have no zero tick, interval label,
or separate reset button; right-click resets. Active angular suffixes switch rad/deg and passive
units remain neutral. `make ui-diagnostics` exercises these production paths.

## Final editor acceptance

UI authoring now defaults to deferred model edits. A yellow viewport border and interactive
Apply/Discard hint accompany pending declarations; repeated property writes coalesce. One final
composition is installed in the existing worker queue. Workspace history records a grouped edit;
standalone adapters retain atomic failure recovery without acquiring new undo capabilities. Camera
navigation, supported direct properties and public Session/RPC calls preserve their direct behavior.
Settings can restore realtime UI editing. View Camera transitions ease and hand off from the visible
pose; tracking has one-click stop/resume. Normal property rows are transparent while selected rows
retain padding.

Final verification: 1,699 fast, 149 integration, 427 physics tests (2 skips), MuJoCo audit and adapter
conformance; 200 focused UI cases per backend on bgfx/OpenGL/wgpu. English/Chinese 100%/150% layout
captures passed containment checks and were inspected. Documentation checks passed.

On the Apple M5, `actuator_visuals` at 1600×1000 with VSync off, 240 samples per variant and 180
widget timing samples: populated diagnostics plus pending Apply had frame median/P95 3.005/3.313 ms
on bgfx and 4.783/8.072 ms on OpenGL. Direct capsule/hint drawing was 0.259/0.279 ms. A visible
48-snapshot timeline drew in 0.379/0.397 ms per frame; full frame median/P95 was 3.324/7.167 ms and
4.841/7.851 ms. Geometry caches had zero warmed misses. Sixty dimension writes coalesced into one
command, median staging 0.013 ms, with 1.50–1.64 ms Apply CPU time in that small scene. These are
CPU submission measurements, not display latency or a guarantee for large models.

An existing limitation remains for entirely anonymous attached-body ownership in placement
previews/state migration; identifiable body names avoid that ambiguity. The two strict MSAA corpus
failures and large-model long-frame limits above remain explicitly unresolved.
