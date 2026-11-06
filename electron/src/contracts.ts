import type * as THREE from 'three';
export type EntityKind =
  | 'robot'
  | 'world'
  | 'environment'
  | 'geom'
  | 'link'
  | 'joint'
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'capsule'
  | 'ground'
  | 'light'
  | 'camera';
export type Tool = 'select' | 'position' | 'rotation' | 'scale' | 'joint';
export type GeometryMode = 'default' | 'visual' | 'collision' | 'both';
export type Frame = 'world' | 'body';
export interface SceneNode {
  id: string;
  name: string;
  kind: EntityKind;
  object: THREE.Object3D;
  parentId?: string;
  visible: boolean;
  properties?: Record<string, string | number>;
}
export interface Joint {
  id: string;
  name: string;
  type: string;
  axis: THREE.Vector3;
  lower: number;
  upper: number;
  value: number;
  object: THREE.Object3D;
  origin: THREE.Matrix4;
  parent: string;
  child: string;
  mimic?: { joint: string; multiplier: number; offset: number };
  position?: THREE.Vector3;
  rotation?: THREE.Quaternion;
  angularLimit?: number;
  bodyId?: string;
}
export interface ProjectFile {
  name: string;
  path: string;
  kind: 'file' | 'directory';
  children?: ProjectFile[];
}
export interface SceneDocument {
  name: string;
  root: THREE.Group;
  nodes: SceneNode[];
  joints: Joint[];
  files: ProjectFile[];
  source?: string;
  warnings: string[];
}
export interface OpenResource {
  name: string;
  path: string;
  content: string;
  resourceBase: string;
  files: ProjectFile[];
}
export interface NativeMenuState {
  focusKey?: string | null;
  focusAllKey?: string | null;
  locale: 'en' | 'zh-CN';
  theme: 'dark' | 'light';
  canUndo: boolean;
  canRedo: boolean;
  hasSelection: boolean;
  hasProject?: boolean;
  showLeft: boolean;
  showInspector: boolean;
  showTimeline: boolean;
  sceneId?: string;
  textInput?: boolean;
}
export interface FileEntry {
  name: string;
  path: string;
  directory: boolean;
}
export interface FileListing {
  path: string;
  parent: string | null;
  entries: FileEntry[];
  shortcuts: { label: string; path: string }[];
  truncated: boolean;
}
export interface DesktopBridge {
  browseFiles: (path?: string, includeHidden?: boolean) => Promise<FileListing>;
  openModelPath: (path: string) => Promise<OpenResource>;
  openUrdfPath: (path: string) => Promise<OpenResource>;
  openProjectPath: (path: string) => Promise<{ path: string; name: string; files: ProjectFile[] }>;
  readProjectDirectory: (path: string) => Promise<ProjectFile[]>;
  searchModelAssets: (path: string, query: string) => Promise<{ files: ProjectFile[]; truncated: boolean }>;
  searchProject: (
    path: string,
    query: string,
  ) => Promise<{ files: ProjectFile[]; truncated: boolean }>;
  openScenePath: (path: string) => Promise<{ content: string; path: string }>;
  saveBinaryFile: (path: string, content: Uint8Array, overwrite?: boolean) => Promise<string>;
  saveScenePath: (path: string, content: string, overwrite?: boolean) => Promise<string>;
  confirmClose: () => Promise<void>;
  hasNativeMenu: boolean;
  readClipboardText: () => Promise<string>;
  writeClipboardText: (text: string) => Promise<void>;
  onMenuAction: (callback: (action: string) => void) => () => void;
  updateMenuState: (state: NativeMenuState) => Promise<void>;
  readUrdf: (path: string) => Promise<OpenResource>;
  platform: string;
}
declare global {
  interface Window {
    mojive?: DesktopBridge;
  }
}
