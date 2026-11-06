import { capturePose, type Keyframe } from '../editor';
import type { SceneDocument } from '../contracts';
import { bindSceneCommands } from '../scene-commands';
import type { ModelBundle } from '../resources/modelBundle';
import type { PhysicsBackendFactory, PhysicsSnapshot } from './contracts';
import { SceneBinding } from './SceneBinding';
import { SimulationSession } from './SimulationSession';

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

  static async load(bundle: ModelBundle, factory: PhysicsBackendFactory): Promise<PhysicsDocument> {
    const backend = await factory(bundle), session = new SimulationSession();
    try {
      await session.attach(backend);
      return new PhysicsDocument(bundle, new SceneBinding(backend.source, session.current.frame!), session);
    } catch (error) { session.dispose(); throw error; }
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
