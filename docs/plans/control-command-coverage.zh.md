# CLI/RPC 命令边界与缺口

2026-09-13 对照 `python/commands.py`、`python/control/operations.py` 与
`python/control/application.py`。这是任务交接时的实现清单；运行时可发现接口仍以
`mojive operations --json` 为准。不要通过给每个 Python 类自动生成 RPC 的方式扩大接口。
CLI `control` 与 socket RPC 使用相同操作目录、输入校验和 Session 路由。

## 已公开的操作

| Python 命令 | CLI/RPC 操作 | 责任边界 |
| --- | --- | --- |
| `Pause`、`Play`、`Step`、`Reset`、`SetSpeed` | `pause`、`resume`、`step`、`reset`、`set_speed` | 仿真时钟能力；外部时钟所有者限制仍由操作和 Session 检查 |
| `LoadKeyframe` | `set_keyframe` | 加载已有关键帧，不等于修改关键帧定义 |
| `SetQpos` | `set_qpos` | 单项调用命令；完整向量通过状态恢复边界 |
| `SetCtrl`、`SetCtrlVector` | `set_ctrl` | 标量与完整控制向量分别执行；向量先整体验证 |
| `LoadAsset`、`Reload`、`NewScene`、`OpenScene`、`SaveScene` | `load`、`reload`、`new_scene`、`open_scene`、`save_scene` | 文档替换与序列化；不是可嵌套的编辑事务 |
| `Select`、`SelectNode`、`SetVisible`、`SetVisualGroup` | `select_object`、`select_node`、`set_visible`、`set_visual_group` | 选择与显示状态；不伪装成几何模型写入 |
| `SetPose`、`SetScale` | `set_pose`、`set_scale` | 调用既有目标解析与能力边界；Scale 立即烘焙为尺寸并恢复 identity |
| `SetGeometryColor`、`SetGeometrySize` | `set_geometry_color`、`set_geometry_size` | 颜色使用 Session 合成状态；尺寸使用已有局部几何约定 |
| `AddSceneObject`、`RemoveSceneObject` | `add_scene_object`、`remove_scene_object` | 中立场景对象生命周期；创建返回真实 ID |
| `DuplicateSceneEntity`、`RemoveSceneEntity`、`RenameSceneEntity` | `duplicate_scene_entity`、`remove_scene_entity`、`rename_scene_entity` | 中立实体生命周期 |
| `AddSceneCamera`、`SetSceneCamera`、`RemoveSceneCamera` | `add_scene_camera`、`set_scene_camera`、`remove_scene_camera` | 场景相机实体，不是 viewport 临时相机 |
| `AddSceneLight`、`RemoveSceneLight` | `add_scene_light`、`remove_scene_light` | 场景光源实体生命周期 |
| `SetModelSource` | `mujoco.set_model_source` | 显式 MuJoCo/MJCF 能力，不冒充后端中立编辑 |
| `Undo`、`Redo` | `undo`、`redo` | Session 文档历史 |

## 内部事务与本地交互

| Python 命令 | 当前接口与理由 |
| --- | --- |
| `BeginEditTransaction`、`EndEditTransaction`、`CancelEditTransaction` | 由 `edit_scene` 成对管理。禁止客户端跨请求悬挂事务；断连不能留下未完成编辑。 |
| `PreviewSceneModelTransform`、`ClearSceneModelTransformPreview` | 本地拖动预览，由 UI 生命周期清理；远程提交应使用最终模型变换操作。 |
| `Perturb`、`ClearPerturb` | 物理鼠标交互，需要持续持有/释放的输入生命周期；现有无状态 RPC 不具备该租约。 |
| `SetCamera` | 本地交互相机命令。远程使用明确分开的 `set_viewport_camera` 与 `set_capture_camera`；不可混用 scene camera ID。 |
| `SetQposBatch` | 本地关节批编辑；远程已提供完整向量的 `set_qpos`。稀疏批更新需要另外定义重复索引、原子性与错误契约。 |

## 尚未公开的功能

这些是明确保留的缺口，不代表功能已由其他操作完整覆盖。新增接口时先检查能力发现、
文档前置条件、目标身份、事务原子性与保存重开，再定义后端中立或能力专属的操作名。

| Python 命令 | 后续工作与当前未公开原因 |
| --- | --- |
| `AddSceneModel`、`RemoveSceneModel`、`SetSceneModelTransform` | 多模型组合的稳定模型身份、重编译与资源所有权需要独立 schema；`load` 只替换整个文档。 |
| `AddResourceRoot`、`RemoveResourceRoot` | 资源路径影响后续解析；需要明确相对路径基准与文档持久化范围。 |
| `AddModelElement`、`DuplicateModelElement`、`RemoveModelElement`、`RenameModelElement`、`ModelEditBatch` | 拓扑编辑及创建后引用依赖 batch-local identity。不能把临时节点索引直接暴露为稳定身份。 |
| `AddModelComponent`、`UpdateModelComponent`、`RemoveModelComponent` | 能力专属模型组件；需要声明合法组件类型和属性 schema。 |
| `AddModelKeyframe`、`SetModelKeyframe`、`RemoveModelKeyframe` | 模型关键帧写入；需要模型身份与各状态向量维度校验。 |
| `SetJointProperties`、`SetJointAdvancedProperties`、`SetSiteProperties`、`SetBodyProperties` | 物理/模型源属性；需要明确只读字段、单位和重建范围。 |
| `SetGeometryProperties`、`SetGeometryAdvancedProperties`、`SetGeometryShape` | 后端模型几何属性与拓扑；现有尺寸/颜色操作只覆盖中立的窄接口。 |
| `ImportModelGeometryResource`、`ImportModelAsset`、`ImportModelTexture`、`SetHeightFieldSize` | 外部资源导入、路径解析与引用验证；不能把文件失败留到部分写入之后。 |
| `RenameModelAsset`、`DuplicateModelAsset`、`ReplaceModelAssetFile`、`RemoveModelAsset` | 模型资产依赖关系与重建；需要检查资产被引用时的行为。 |
| `CreateModelMaterial`、`AddModelMaterial`、`SetGeometryMaterial` | 模型源材质与几何绑定；需要稳定资产引用，不能复用渲染帧材质数组索引。 |
| `SetLight`、`SetEnvironment`、`SetSkybox`、`SetMaterial` | 场景外观写入仍有缺口；需区分实体 ID、纹理资源名和结构版本内数组索引。 |
| `SetEqualityEnabled` | 物理约束能力专属写入；需要可发现的约束身份和状态语义。 |
| `StepBack` | 本地显示帧历史；远程需说明与物理步进、状态恢复的区别。 |
| `StartStateTakeRecording`、`StopStateTakeRecording`、`PlayStateTake`、`PauseStateTake`、`SeekStateTake`、`SetStateTakeLoop`、`ClearStateTake` | 临时 take 的生命周期与帧索引，当前没有对应 RPC schema；与模型关键帧、文件录屏均不同。 |
| `CaptureSceneSnapshot`、`RestoreSceneSnapshot`、`RemoveSceneSnapshot` | Session 临时快照身份与兼容性校验；现有 capture 图像接口不覆盖物理快照。 |

## 验证与后续边界

Scale 的 schema、目标解析、烘焙、溢出/下溢、Undo/Redo、保存重开与真实 CLI/socket 链路
由 `tests/test_rpc_scale.py` 覆盖；`tests/gpu/test_control_scale.py` 验证 attached viewer
中真实像素随缩放变化，撤销后恢复。`make agent-control` / `make agent-viewer` 共用
`examples/agent_inspection.py`，在同一编辑事务中执行绝对尺寸后再缩放。

当前 `inspect_object` 与既有界面一样读取 Session 合成状态；有未提交预览时，返回预览
尺寸和 Scale。文档修改通过 `writes_document` 标记统一拒绝未提交 UI 草稿；查询、选择
和显示可见性仍可用。RPC 与 UI Apply 通过 `session/edit_batch.py` 共用执行器，提交前
检查文档版本，按操作分组重建，保留 UI 创建后的真实身份绑定和失败回滚证据。
具体验证入口见 [RPC 控制指南](../how-to/rpc-control.md)及测试矩阵。
本表不把“一个 Undo 记录”当作“只有一次后端重建”，也不宣称全部 CLI/RPC 缺口已完成。
