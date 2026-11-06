import type { ModelBundle } from '../resources/modelBundle';

export type Vector3 = [number, number, number];
/** Neutral rotations use xyzw quaternions and all scene coordinates are Z-up. */
export type Quaternion = [number, number, number, number];
export interface Transform { position: Vector3; rotation: Quaternion }
export type PhysicsCapability = 'environment' | 'controls' | 'jointPose' | 'perturbation' | 'state';

export type GeometryShape =
  | { type: 'plane' | 'box' | 'ellipsoid'; size: Vector3 }
  | { type: 'sphere'; radius: number }
  | { type: 'cylinder' | 'capsule'; radius: number; length: number }
  | { type: 'mesh'; positions: Float32Array; normals?: Float32Array; uv?: Float32Array; indices?: Uint32Array }
  | { type: 'heightfield'; rows: number; columns: number; size: [number, number, number, number]; heights: Float32Array };

export interface PhysicsBody extends Transform { id: string; name: string; parentId: string | null; dynamic: boolean }
export interface PerturbationTarget extends Transform {
  bodyId: string;
  mode: 'translation' | 'rotation';
  localPosition: Vector3;
  strength: number;
}
export interface PhysicsGeometry extends Transform {
  id: string;
  name: string;
  bodyId: string;
  shape: GeometryShape;
  color: [number, number, number, number];
  metallic: number;
  roughness: number;
  group: number;
  collidable: boolean;
  role?: 'visual' | 'collision' | 'both';
  defaultVisible?: boolean;
  deformable?: boolean;
  material?: { name: string; texture: number; repeat: [number, number]; uniform: boolean; emission: number; specular: number; shininess: number; reflectance: number };
}
export interface PhysicsTexture {
  name: string; type: '2d' | 'cube' | 'skybox'; width: number; height: number; channels: number; pixels: Uint8Array;
}
export interface PhysicsVisualEnvironment {
  ambient: Vector3; headlight: Vector3; fog: Vector3; fogStart: number; fogEnd: number;
  near: number; far: number; extent: number;
}
export interface PhysicsJoint {
  id: string;
  name: string;
  bodyId: string;
  type: 'revolute' | 'continuous' | 'prismatic' | 'ball' | 'free';
  axis: Vector3;
  anchor: Vector3;
  limits?: [number, number];
  /** Offset in frame.jointPositions; scalar, xyzw, or xyz + xyzw. */
  offset: number;
  width: 1 | 4 | 7;
}
export interface PhysicsActuator {
  id: string;
  name: string;
  jointId?: string;
  range?: [number, number];
}
export interface PhysicsCamera extends Transform {
  id: string; name: string; bodyId: string; fov: number; orthographic: boolean;
  resolution: [number, number];
}
export interface PhysicsLight extends Transform {
  id: string; name: string; bodyId: string; type: 'directional' | 'point' | 'spot'; direction: Vector3;
  color: Vector3; cutoff: number;
  active?: boolean; castShadow?: boolean; ambient?: Vector3;
}
export interface PhysicsSource {
  id: string;
  name: string;
  backend: string;
  version: string;
  capabilities: PhysicsCapability[];
  bodies: PhysicsBody[];
  geometries: PhysicsGeometry[];
  joints: PhysicsJoint[];
  actuators: PhysicsActuator[];
  cameras: PhysicsCamera[];
  lights: PhysicsLight[];
  environment: EnvironmentField[];
  warnings: string[];
  textures?: PhysicsTexture[];
  visual?: PhysicsVisualEnvironment;
  keyframes?: { id: string; name: string; time: number; frame: PhysicsFrame; snapshot: PhysicsSnapshot }[];
}
export interface PhysicsFrame {
  sourceId: string;
  time: number;
  /** World transforms, in source.bodies order: xyz + xyzw. */
  bodyTransforms: Float64Array;
  /** World anchors and axes, xyz + xyz, in source.joints order. */
  jointFrames: Float64Array;
  /** World camera poses, xyz + xyzw, including tracking cameras. */
  cameraTransforms: Float64Array;
  /** World light positions and directions, xyz + xyz. */
  lightFrames: Float64Array;
  jointPositions: Float64Array;
  controls: Float64Array;
  geometryUpdates?: { id: string; positions: Float32Array; normals: Float32Array; count: number }[];
}

export type EnvironmentValue = number | boolean | number[];
export type EnvironmentValues = Record<string, EnvironmentValue>;
interface EnvironmentFieldBase {
  id: string; label: string; group: string; advanced?: boolean;
}
export type EnvironmentField = EnvironmentFieldBase & (
  | { type: 'number'; min?: number; max?: number; integer?: boolean; step?: number; unit?: string }
  | { type: 'vector'; size: number; min?: number; max?: number; step?: number; unit?: string }
  | { type: 'enum'; choices: { value: number; label: string }[] }
  | { type: 'boolean' }
);

/** Opaque engine state. The backend owns its encoding and compatibility checks. */
export interface PhysicsSnapshot {
  sourceId: string;
  backend: string;
  format: string;
  state: number[];
  environment: EnvironmentValues;
  /** Backend-owned transient interaction state needed for exact continuation. */
  interaction?: unknown;
}
export type PhysicsCommand =
  | { type: 'step'; count: number }
  | { type: 'reset' }
  | { type: 'controls'; values: { id: string; value: number }[] }
  | { type: 'jointPose'; id: string; position: number[] }
  | { type: 'perturb'; bodyId: string; force: Vector3; torque: Vector3 }
  | { type: 'perturbTarget'; target: PerturbationTarget }
  | { type: 'clearPerturbation' }
  | { type: 'environment'; values: EnvironmentValues }
  | { type: 'resetEnvironment' }
  | { type: 'restore'; snapshot: PhysicsSnapshot };

export interface PhysicsBackend {
  readonly source: PhysicsSource;
  readonly defaults: EnvironmentValues;
  frame(): Promise<PhysicsFrame>;
  environment(): Promise<EnvironmentValues>;
  execute(command: PhysicsCommand): Promise<PhysicsFrame>;
  capture(): Promise<PhysicsSnapshot>;
  dispose(): void;
}
export type PhysicsBackendFactory = (bundle: ModelBundle) => Promise<PhysicsBackend>;
