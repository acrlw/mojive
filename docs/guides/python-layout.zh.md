# Python 模块地图

`python/src/mojive` 按职责组织实现。根目录保留 `types.py`、`commands.py`、
`math3d.py` 等共享契约，以及已发布导入路径的薄兼容入口。修改行为时进入实际实现包，
不要在兼容入口添加业务逻辑，也不要通过模块别名或重复转发方法隐藏依赖。

| 包 | 负责什么 | 主要入口 |
|---|---|---|
| `application/` | 组装 Viewer、选择后端、宿主生命周期、旧 Renderer API | `composition.py`、`backends.py`、`passive.py`、`renderer.py` |
| `scene/` | 场景实体、资源定位、边界与几何查询、文件读写 | `model.py`、`assets.py`、`bounds.py`、`geometry.py`、`io.py`、`workspace.py`、`state.py` |
| `session/` | 文档状态、选择、编辑事务、播放、结构刷新 | `core.py`、`editing.py`、`playback.py`、`source.py` |
| `session/dispatch/` | 将类型化命令交给对应的处理函数 | `documents.py`、`transforms.py`、`properties.py`、`assets.py`、`physics.py`、`playback.py` |
| `control/` | 可发现的操作、参数校验、应用控制、RPC 传输 | `operations.py`、`schema.py`、`application.py`、`rpc.py` |
| `capture/` | 图像与录制契约、共享图像、快照和视频写入 | `types.py`、`shared_image.py`、`recording.py` |
| `remote/` | 场景快照发布、接收、命令桥接 | `protocol.py`、`publisher.py`、`adapter.py`、`commands.py`、`bridge.py` |
| `interaction/` | 输入契约、与 UI 无关的 gizmo 几何和命中计算 | `input.py`、`gizmo.py` |
| `drawing/` | 与绘制后端无关的二维曲线、多边形和 Canvas 契约 | `curves.py`、`polygons.py`、`drag_link.py`、`canvas.py` |
| `adapters/` | 场景与物理后端适配 | `base.py`、`mujoco/`、`static/`、`toy/` |
| `render/` | 渲染契约、后端与离屏渲染 | `backend.py`、`offscreen.py`、`opengl/`、`webgpu/`、`native/` |
| `cli/` | 命令行解析及命令实现 | `parser.py`、`inspection.py`、`viewer.py`、`control.py`、`capture.py` |
| `tools/` | 可执行的诊断、性能测量和验收场景 | 相应 Make target 指向的模块 |

## 状态与命令边界

`Session` 是文档状态的唯一所有者；adapter 是物理状态的所有者。`session/editing.py`、
`playback.py`、`source.py` 中的私有实现类只是同一个 Session 的方法分组，不独立持有状态，
也不是后端扩展接口。MuJoCo adapter 和大型 UI 控制器采用同样的约束：构造和销毁集中在
`core.py` 或 `adapter.py`，功能模块之间通过所有者的明确方法协作。

`Session.submit()` 负责事务和结果处理，随后通过 `session/dispatch/` 的命令表查找处理函数。
处理函数直接操作所属 Session，并在写入边界检查 adapter 能力。新增命令时更新命令契约、
对应领域的处理函数和路由表；不要恢复一个持续增长的 `isinstance` 分支链。命令子类仍按
原有匹配顺序解析，未知命令仍返回失败结果。

`control/operations.py` 定义 RPC 与 CLI 共用的操作目录，`schema.py` 负责校验，
`rpc.py` 仅处理传输和请求调度。导入 RPC 客户端或离线查询操作目录，不能提前创建应用、
加载物理引擎或初始化图形环境。`control/__init__.py` 按需导出应用对象来保持这条边界。

## UI 中到哪里修改

| 功能 | 实现位置 |
|---|---|
| 主循环、初始化、退出 | `ui/app/core.py` |
| 模型加载、编辑预览、资源选择 | `ui/app/loading.py`、`model_edits.py`、`resource_dialogs.py` |
| 输入、相机导航、精确 gizmo 输入 | `ui/app/input.py`、`navigation.py`、`gizmo_input.py` |
| 菜单、viewport、状态栏、录制 | `ui/app/menus.py`、`viewport.py`、`status.py`、`capture.py` |
| gizmo 状态、投影、拖拽、范围、辅助线 | `ui/gizmo/core.py`、`projection.py`、`dragging.py`、`joint_ranges.py`、`guides.py` |
| gizmo 绘制与目标解析 | `ui/gizmo/drawing.py`、`targets.py` |
| Inspector 属性分类 | `ui/panels/inspector/model.py`、`transform.py`、`physics.py`、`geometry.py`、`environment.py` |
| Inspector 复用字段布局 | `ui/panels/inspector/fields.py` |
| viewport 胶囊、图标、输入提示、状态信息 | `ui/viewport_widgets/capsules.py`、`glyphs.py`、`hints.py`、`status.py` |
| 图标设计参数、生产图标、预计算布局 | `ui/icons.py`、`ui/icon_presets.json` |

私有方法分组之间不重复定义同名方法，不额外增加构造、资源释放或权威状态。
只有调用者确实需要独立生命周期时，才提取成独立对象。避免仅为了满足行数限制而拆文件；
现有独立、单一职责的中型模块不需要再增加包装层。

## 性能与验收

生产图标从 `ui/icon_presets.json` 读取与当前参数匹配的布局和测量结果，减少首帧的几何拟合。
Icon Library 继续动态计算可调参数。修改图标默认参数后运行 `make icon-presets`，通过
相关图标测试核对预计算结果；参数不匹配时走原始计算路径，不使用过期结果。

使用以下入口测量启动时间：

```bash
make startup-profile ARGS="--backend opengl wgpu bgfx --repeats 3 --compare-icons"
```

每次试验使用新的 Python 进程，交替测量顺序，复制当前布局到输出目录，并在四帧后退出。
报告分别记录进程启动到首次 `end_frame()` 返回、窗口显示到该次返回的时间。
它们不是 GPU 完成或屏幕扫描时间；操作系统和驱动缓存也未清空。截图在计时区间之外。
`--compare-icons` 仅在测量进程中关闭预计算布局作为对照；`--profile` 生成调用栈统计，
其额外开销不应混入正常启动速度结论。

验证要求以[测试矩阵](testing.md#change-mapping)为准。常用入口为 `make check`、
`make test-physics`、`make gpu`、`make gpu-wgpu`、`make native-features-test` 和
`make docs-check`。GPU 与窗口验收串行运行。性能比较也应在这些测试结束后单独进行。
公共导入兼容测试在 `test_package_layout.py`，依赖边界在 `test_layering.py`；
针对内部函数的测试监测点应指向实际实现模块，不能只修改兼容入口的同名导出。
