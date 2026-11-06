import { capturePose, type Keyframe } from '../editor';
import type { SceneDocument } from '../contracts';
import { bindSceneCommands } from '../scene-commands';
import type { ModelBundle } from '../resources/modelBundle';
import type { PhysicsBackendFactory, PhysicsSnapshot } from './contracts';
import { SceneBinding } from './SceneBinding';
import { SimulationSession } from './SimulationSession';
import { checkModelLoad, type ModelLoadOptions } from '../resources/loading.ts';
import { disposeDocument } from '../scene-resources.ts';

/** Composition boundary between simulation ownership and the editable viewer document. */
export class PhysicsDocument {
  readonly session: SimulationSession;
  readonly bundle: ModelBundle;
  readonly binding: SceneBinding;
  readonly keyframes: Keyframe[];
  private unbind: () => void = () => {};
  private unsubscribe: () => void;
  private onError: (error: unknown) => void = () => {};

  private constructor(bundle: ModelBundle, binding: SceneBinding, session: SimulationSession) {
    this.bundle = bundle; this.binding = binding; this.session = session;
    this.keyframes = (binding.source.keyframes ?? []).map((key) => {
      binding.apply(key.frame);
      return { id: key.id, name: key.name, time: Math.max(0, key.time), pose: capturePose(binding.document), simulation: key.snapshot };
    });
    binding.apply(session.current.frame!);
    this.bind();
    let previous = session.current.frame;
    this.unsubscribe = session.subscribe(({ frame }) => {
      if (frame && frame !== previous) { binding.apply(frame); previous = frame; }
    });
  }

  static async load(bundle: ModelBundle, factory: PhysicsBackendFactory, options?: ModelLoadOptions): Promise<PhysicsDocument> {
    checkModelLoad(options);
    const backend = await factory(bundle, options), session = new SimulationSession();
    let binding: SceneBinding | undefined;
    let attaching = false;
    try {
      checkModelLoad(options);
      attaching = true;
      await session.attach(backend);
      binding = await SceneBinding.create(backend.source, session.current.frame!, options);
      checkModelLoad(options);
      return new PhysicsDocument(bundle, binding, session);
    } catch (error) {
      if (!attaching) backend.dispose();
      session.dispose();
      if (binding) disposeDocument(binding.document);
      throw error;
    }
  }
  get document(): SceneDocument { return this.binding.document; }
  setErrorHandler(handler: (error: unknown) => void): void { this.onError = handler; }
  private bind(): void {
    this.unbind();
    this.unbind = bindSceneCommands(this.document, {
      setJointPosition: (id, position) => {
        void this.session.execute({ type: 'jointPose', id, position }).catch((error) => this.onError(error));
      },
      ...(this.binding.source.capabilities.includes('perturbation') ? { perturbation: {
        bodyAt: (id: string) => this.binding.bodyAt(id),
        update: (target: import('./contracts').PerturbationTarget) => {
          void this.session.execute({ type: 'perturbTarget', target }).catch((error) => this.onError(error));
        },
        clear: () => { void this.session.execute({ type: 'clearPerturbation' }).catch((error) => this.onError(error)); },
      } } : {}),
    });
  }
  async restore(document: SceneDocument, snapshot: PhysicsSnapshot): Promise<void> {
    this.session.pause();
    this.binding.validateDocument(document);
    await this.session.execute({ type: 'restore', snapshot });
    this.binding.rebind(document);
    this.binding.apply(this.session.current.frame!);
    this.bind();
  }
  dispose(): void { this.unbind(); this.unsubscribe(); this.session.dispose(); }
}
