# Mojive Electron 架构

这个目录是独立的 Electron 实现，不改变 Python/ImGui 版的默认行为，也不连接 MuJoCo。

## 所有权

- `main/main.cjs` 负责窗口、固定文件/剪贴板 IPC 与受限资源协议。Renderer 不具有 Node.js 权限。`src/FileDialog.tsx` 负责自定义文件浏览、选择和覆盖确认；不调用系统文件对话框。
- `main/menu.cjs` 负责 macOS 系统菜单、原生快捷键和菜单状态；`preload.cjs` 提供固定 IPC 接口。
- `src/scene.ts` 负责 URDF、资源加载、关节运动学、实体创建和删除。失败的资源加载不替换当前文档。
- `src/SceneCatalog.ts` 提供示例目录；其中的 joint scene 从随应用打包的 `assets/joint_gizmo.xml` 副本构建几何和运动学。
- `src/JointPanel.tsx` 与 `JointEditor.tsx` 共用关节行、数值字段和单事务滑轨；`joint-pose.ts` 将 free 控件的世界姿态映射回内部局部配置。
- `src/contracts.ts` 定义 UI、场景、viewport 与桌面边界之间的契约。
- `src/App.tsx` 管理 selection、历史记录、面板、文档打开/保存和命令事务。
- `src/editor.ts` 保存场景与姿态，分别插值 authored joint origin、标量 joint value 和多自由度配置。
- `src/Viewport.tsx` 管理 camera、渲染生命周期与指针所有权；`viewport/` 保留从原版移植的 shader、轮廓、渐隐和交互。
- `src/Timeline.tsx` 是本地姿态动画编辑器。时间轴不依赖模拟器；保存的数据不冒充物理 rollout。

`SceneDocument.root` 始终是无选择身份的场景集合。world、robot、link、geom、joint、camera 和 light 各有自己的 object ID。URDF robot 和用户添加的对象可以独立存在；删除 robot 不会删除场景容器。

## 示例目录与运动学

| Scene ID            | 内容                                                                                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `joint_gizmo`       | 原生 joint acceptance scene：45 个实体、19 个 geom、9 个 joint，包括 revolute、prismatic、ball、free、同一 body 的三个 direct joint，以及 ±2° / ±0.005 m 精密限位 |
| `atelier_arm`       | URDF 机械臂、联动夹爪和 transmission/actuator 元数据；visual/collision 都是可检查的 geom 节点                                                                     |
| `geometry_lab`      | 可视几何、碰撞代理及重叠表面的显示对比                                                                                                                            |
| `primitive_gallery` | 基本体、地面、灯光和相机                                                                                                                                          |

`createCatalogScene(id)` 返回完整 `SceneDocument`。`createNativeJointScene(xml)` 只适配该维护示例所用的几何和运动学子集，不是通用 MJCF 或 MuJoCo 加载器。示例 XML 保留原有 body、joint、material 与 camera 名称。

标量 joint 用 `setJoint(doc, id, value)` 修改，内部长度单位为米、角度为弧度。ball/free joint 额外保存 `position: Vector3` 和 `rotation: Quaternion`，通过 `setJointConfiguration(doc, id, { position, rotation })` 修改；输入 quaternion 顺序为 `x, y, z, w`。配置始终相对 `joint.origin`。ball 不允许平移，存在 `angularLimit` 时按旋转角幅值限制；free 可独立平移和旋转。界面可以显示角度，但必须在命令边界转换为弧度或 quaternion。free 数值控件、折叠摘要和 Copy state 显示世界位置/旋转，与原生 qpos 和 viewport 操作一致；转换通过 authored origin 与 parent world matrix 的逆变换完成，保存格式仍使用局部配置。ball 始终显示 joint-local 姿态。

多个 direct joint 通过连续的运动 Group 作用于一个 link，`joint.bodyId` 标识共同的 body；每个 joint 仍有独立 object ID。viewport chooser 选择当前 joint，不能以同一个 child 名称替代 joint 身份。`authorJointTransform` 在直接编辑 joint 节点时从当前 transform 中扣除运动配置，再更新 authored origin。

## 几何与文档

每个 URDF visual/collision placement 是独立 geom 节点。节点外层 Group 的 `visible` 由用户控制；内层 `roleGroup` 保存源文件的 visual/collision 显示成员关系。Default 保留源显示状态和材质，Visual/Collision/Both 只在渲染期间切换角色层，不能覆盖用户隐藏状态。Both 分别生成最近表面的图层再合成，避免普通透明对象排序造成的重复叠色。

Three.js 的 parametric geometry 会按原始参数序列化，忽略已经烘焙的顶点旋转。所有转为 Z-up 的 cylinder/capsule 都在旋转后复制为中性的 `BufferGeometry`，以便 Undo、保存及姿态回放保留精确顶点和法线。

`.mojive.json` 保存场景图、节点身份、joint origin/value/configuration、资源元数据和关键帧。贴图通过 Three.js 导出的内嵌图像保存。加载先验证有限数值、完整 pose、唯一身份、真实父子关系、joint/link 连通性、mimic 依赖和 actuator 引用，再构建替换文档；非法文件不会先清空当前场景。

`WorkspaceTimeline` 分别保存命名 keyframes、`SceneSnapshot` 和 `SceneTake`。snapshot 保存一次完整姿态，take 保存严格递增时间戳的完整姿态帧；`saveWorkspace` 的可选第四参保存 snapshots/takes/activeTakeId，旧文件没有这些字段时读取为空。取帧采用原生二分规则：scrub 选最近帧，等距选前帧；playback 使用最后一帧 `time <= cursor`，不插值 take。snapshot/take 应用前比较完整 object/joint/origin/configuration 身份集合，结构不兼容时先报错，不部分覆盖当前场景。旧快照仍可随项目保存供检查或删除。

关节卡片使用 outer 12、padding 8、inner 4 的普通圆角，数值控件与 Inspector 共用。scalar slider 对照 `ui/panels/value_cards.py`、`ui/value_rail.py` 与 `workspace/design.py`：4-point 圆头轨道、12-point 圆手柄、从起点填充至手柄、没有 viewport precision rail 的限位竖线；不同主题的 idle/hover/held ink 使用原始语义色。

每次完成的数值或拖动编辑对应一个 Undo 事务；失败的命令恢复其前置快照。历史记录只存在于当前会话。`removeNode(doc, id)` 在 incoming joint 边界删除 body 子树，同时删除依赖它的 mimic 分支和失效 actuator 元数据。复合 joint body 的整个运动链一起删除，避免悬空引用。只释放不再被其他对象共享的几何、材质与贴图。旧关键帧中已删除对象的通道保留为无作用数据，不会阻止其余姿态恢复。

## 系统菜单与桌面边界

macOS 使用系统菜单；`hasNativeMenu` 为真时 Renderer 不再显示重复的应用菜单。Windows/Linux 使用 Renderer 菜单，命令 ID 相同。菜单向 Renderer 发送 `file.*`、`edit.*`、`add.*`、`scene.*`、`view.*`、`theme.*`、`locale.*`、`app.settings`、`app.about` 和 `help.interaction`，不直接改场景。

Renderer 通过 `onMenuAction(callback)` 订阅命令，卸载时调用返回的 unsubscribe。`updateMenuState` 传递 locale、theme、Undo/Redo 可用性、selection、面板显示、scene ID 和 text-input focus。中文/英文与深色/浅色菜单由同一状态更新。剪切、复制、粘贴和文本 Undo/Redo 使用原生编辑命令；文本输入拥有焦点时移除 `F`、`Shift+F` 和场景删除的 accelerator，避免抢走输入。

`scripts/compile-menu-icons.cjs` 从相同 SVG masters 输出 16-point 的 1×/2× PNG representation。菜单使用 `nativeImage` template，由 AppKit 控制主题、选中和禁用状态的颜色。Help 的系统搜索由 macOS `help` role 管理，不使用 Web CSS 改造。

`FileDialog` 使用 `browseFiles` 获取目录条目，通过 `openUrdfPath`、`openProjectPath`、`openScenePath` 和 `saveScenePath` 执行明确选择。目录浏览不会授权 renderer 的资源请求；只有选中的资源目录进入 `Projects`。保存先写同目录临时文件并 sync；首次保存通过 hard-link 原子发布，遇到已有文件必须经过应用内覆盖确认；确认后使用 rename 替换。写入失败不会替换旧文件。`beforeunload` 的未保存状态触发 `app.requestClose`，应用自己的 Save/Discard/Cancel 决定是否调用 `confirmClose`。

项目目录以独立的 `{path, name, files}` 状态保存；`openProjectPath` 返回经过 realpath 解析的根路径。打开场景、展开文件夹、关闭项目均不会把资源目录隐式变成项目根。关闭项目只清空目录、文件树与搜索，保留当前场景、未保存状态、时间轴和 Undo/Redo；已加载场景仍可读取原有授权资源。原生 `file.closeProject` 与项目标题栏共用命令，`hasProject` 决定可用性。文件选择器确认时先解析修改后的 Location，不依赖用户提前按 Enter，也不会复用旧目录的选中条目。

`ProjectExplorer` 只在展开时通过 `readProjectDirectory` 加载直接子项，避免先遍历的大目录消耗全局预算、导致其余顶层目录消失。`searchProject` 独立递归搜索未展开目录，返回匹配项及明确的 `truncated` 状态；1,000 条匹配或 100,000 次文件访问达到上限时提示缩小搜索。两个 IPC 均使用已授权目录检查，不开放任意目录内容读取。隐藏文件、依赖/构建目录与符号链接延续原有过滤策略。

`encodeDocument` 在序列化前刷新场景矩阵。Three.js `toJSON` 直接读取缓存矩阵，因此不能依赖下一次 render 更新；同一事件循环内完成的数字编辑也必须形成完整、独立的 Undo transaction。

剪贴板桥只支持最多 4 MiB 的纯文本，组件只能从用户明确的 Copy/Paste 操作调用。所有 IPC 校验主窗口及主 frame。

`mojive-app://` 只提供构建产物；`mojive-resource://` 只读取用户通过对话框打开的目录及其中资源。路径检查在符号链接解析后执行。外部导航、新窗口、未授权 IPC 和通用任意文件读取接口都被阻止。文件浏览器只有目录列表、URDF/scene 读取及 scene 保存这些固定操作。

没有物理适配器时，URDF actuator 仅展示 transmission 元数据，不产生控制力。URDF floating/planar joint 仍明确报错；本地示例中的 ball/free 配置不表示浏览器已经运行 MuJoCo。

## 设计来源

`node scripts/sync-design.mjs` 从现有 design-spec 和 SVG masters 生成 CSS tokens 与图标映射，并复制 MiSans 和许可证。数值字段使用 JetBrains Mono。前端普通 UI 当前统一使用 CSS circular corners；内层半径由 `max(0, outer radius − inset)` 推导，不启用 Figma clip 或全局 squircle。

viewport 连续轮廓与抗锯齿采用原版几何和显式 GPU fringe。普通 UI 圆角与空间 gizmo 的既有 G3 轮廓有不同的所有者，不能通过一个全局 CSS 设置混在一起。

### 场景相机与灯光

`scene-camera.ts` 的 `CameraViewState` 使用世界坐标 eye/target/up、真实投影、垂直 FOV（度）、裁剪面、正交高度、传感器宽度/焦距（毫米）和输出分辨率。`cameraFromView` 构造独立渲染相机，供图片与 Take 导出复用。透视/正交切换保留节点 ID、Three UUID、父级及子对象；输出参数保存在对象的序列化 metadata 中，Undo 和项目保存使用同一份数据。相机视图保持输出宽高比，以取景框标出有效画幅。

`ViewportHandle.useSceneCamera(id, pilot)` 区分场景视图与驾驶：普通场景视图随作者数据变化，导航退出到编辑器；驾驶将轨道/平移/缩放写回相机，每次手势只产生一个事务，Escape 恢复起点。`alignCameraToView`、`createCameraFromView` 和 `captureCameraView` 都操作真实相机；图片按指定分辨率渲染并排除编辑器辅助层。编辑器相机采用有限对数缩放，近裁剪随距离缩小，远裁剪随场景包围球与当前距离变化；显式场景相机保留用户的裁剪参数。

`scene-light.ts` 使用 Three 实际支持的 point/directional/spot/rectangular-area/hemisphere 光源。方向光和聚光的 target 是光源自身的子对象，因此平移/旋转、Undo、保存后恢复保持一致。Native 的独立 diffuse/specular/ambient 及衰减多项式不伪装成无效开关：本实现使用 PBR color/intensity/decay；区域光采用真实矩形宽高。`createViewportScene()` 统一交互与离线导出的环境光、主光、补光和网格。

`viewport/influence.ts` 对照 `python/ui/scene_entities.py`：仅选中相机/光源显示范围，浏览场景相机时隐藏辅助层；相机显示完整裁剪视锥，点光源为三个48段圆环，聚光锥保持180pt轴向/120pt径向预算及近面细化，区域光显示真实发光矩形。遮挡部分使用原版0.28透明度，完整线宽按逻辑像素保持。`layers.helpers` 与 `layers.influence` 控制临时显示，不改变实体 visibility。Both 模式在颜色合成后恢复联合深度，使辅助线的遮挡规则一致。

`CameraPreview` 对照 native `camera_preview.py`：由 Inspector 的 Display 开关明确启用，未锁定时跟随选中相机，Lock 固定相机身份，Pin 固定世界视角及面板位置但继续播放场景。预览可拖动并限制在视口内；窄布局将 Pin/Lock 收进真实操作菜单。透视/正交使用直接二选一的 radio 组，不再要求展开下拉菜单。

`viewport/camera-preview-renderer.ts` 使用独立 WebGL context、场景树、render targets、几何、材质、纹理实例和 shadow maps；只有已解码的图像数据可共享。预览同步作者变换、材质、光源和 geometry mode，不绘制选择高亮、轮廓、辅助线或网格，不改变编辑器 orbit。按 revision、视角及尺寸失效后绘制，隐藏/不可见时不提交帧，关闭释放资源；source 场景的 dispose listeners 不随反复打开累积。

相机 Inspector 的 View/Pilot 动作保持两行两列；当前活动模式的按钮在原位变为 Return to editor，另一模式仍可直接切换，Align/Capture 保持第二行。Projection type 与其余属性共用标签/控件列，两种投影直接等分控件宽度，无内嵌选择胶囊；较窄英文列采用 Persp./Ortho.，完整名称保留在 accessible label 与 tooltip。

相机导航卡片保留两行：名称与 More、投影/分辨率与 View/Pilot。窄栏只在 metadata 内换行；低频 Focus/Align/Capture/Delete 位于支持键盘的菜单。卡片 outer12/inset8/inner4 不变，名称与预览标题的文字横向内边距为8；Inspector 动作在窄栏采用稳定双行文字高度，避免文字溢出或切换模式时布局跳动。
