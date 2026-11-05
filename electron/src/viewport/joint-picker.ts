import type { Joint, SceneDocument, SceneNode } from '../contracts';
import { icons } from '../icons';
import '../smooth-corners';
import './joint-picker.css';

export function jointBody(doc: SceneDocument, selected: string | null) {
  let node = doc.nodes.find((item) => item.id === selected);
  while (node?.kind === 'geom' && node.parentId)
    node = doc.nodes.find((item) => item.id === node!.parentId);
  if (node?.kind === 'joint') {
    const joint = doc.joints.find((item) => item.id === node!.id);
    node = doc.nodes.find(
      (item) => item.id === joint?.bodyId || item.id === joint?.child || item.name === joint?.child,
    );
  }
  return node;
}
export function bodyJoints(doc: SceneDocument, node: SceneNode | undefined) {
  return node
    ? doc.joints.filter(
        (joint) => joint.bodyId === node.id || joint.child === node.id || joint.child === node.name,
      )
    : [];
}
const supported = (joint: Joint) =>
  !joint.mimic && ['revolute', 'continuous', 'prismatic', 'ball'].includes(joint.type);

/** Native contextual chooser: explicit direct joint, remembered per body, dismissal per selection. */
export class JointPicker {
  private document: SceneDocument | null = null;
  private body: string | null = null;
  private dismissed = false;
  private panel: HTMLDivElement | null = null;
  private resize: ResizeObserver | null = null;
  private remembered = new Map<string, string>();
  private click: { selected: string | null; point: [number, number] } | null = null;
  private rows = new Map<string, HTMLButtonElement>();
  private position: [number, number] = [18, 18];
  constructor(
    private host: HTMLElement,
    private callbacks: {
      select: (id: string) => void;
      focus: (id: string) => void;
      restoreFocus: () => void;
      translate: (text: string) => string;
    },
  ) {}
  selectedAt(selected: string | null, point: [number, number]) {
    this.click = { selected, point };
  }
  dismiss() {
    this.dismissed = true;
    this.remove();
  }
  private remove() {
    this.resize?.disconnect();
    this.resize = null;
    this.panel?.remove();
    this.panel = null;
    this.rows.clear();
  }
  private place() {
    if (!this.panel) return;
    this.position[0] = Math.max(
      8,
      Math.min(this.position[0], this.host.clientWidth - this.panel.offsetWidth - 8),
    );
    this.position[1] = Math.max(
      8,
      Math.min(this.position[1], this.host.clientHeight - this.panel.offsetHeight - 8),
    );
    this.panel.style.left = `${this.position[0]}px`;
    this.panel.style.top = `${this.position[1]}px`;
  }
  update(
    doc: SceneDocument,
    selected: string | null,
    active: string | null | undefined,
    enabled: boolean,
  ) {
    if (this.document !== doc) {
      this.document = doc;
      this.remembered.clear();
      this.body = null;
      this.remove();
    }
    const body = jointBody(doc, selected),
      choices = bodyJoints(doc, body),
      explicit = choices.find((joint) => joint.id === selected),
      current = explicit ?? choices.find((joint) => joint.id === active);
    if (current && body) this.remembered.set(body.id, current.id);
    if (this.body !== (body?.id ?? null)) {
      this.body = body?.id ?? null;
      this.dismissed = false;
      this.remove();
      this.position =
        this.click?.selected === selected
          ? [this.click.point[0] + 14, this.click.point[1] + 14]
          : [18, 18];
      this.click = null;
      const remembered = body ? this.remembered.get(body.id) : undefined;
      if (!current && remembered && choices.some((joint) => joint.id === remembered))
        this.callbacks.select(remembered);
    }
    if (!enabled || !body || explicit || choices.length < 2 || this.dismissed) {
      this.remove();
      return;
    }
    if (!this.panel) this.show(body, choices);
    const selectedJoint = current?.id ?? this.remembered.get(body.id);
    for (const [id, row] of this.rows) {
      row.setAttribute('aria-pressed', String(id === selectedJoint));
      row.classList.toggle('active', id === selectedJoint);
    }
  }
  private show(body: SceneNode, choices: Joint[]) {
    const panel = document.createElement('div');
    this.panel = panel;
    panel.className = 'viewport-joint-picker';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', this.callbacks.translate('Choose joint'));
    const heading = document.createElement('div');
    heading.className = 'viewport-joint-picker-heading';
    const title = document.createElement('strong');
    title.textContent = body.name;
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'viewport-joint-picker-close';
    close.setAttribute('aria-label', this.callbacks.translate('Close'));
    close.innerHTML = icons.close;
    close.onclick = () => {
      this.dismiss();
      this.callbacks.restoreFocus();
    };
    heading.append(title, close);
    panel.append(heading);
    let drag: { id: number; x: number; y: number; start: [number, number] } | null = null;
    heading.onpointerdown = (event) => {
      if (event.button !== 0 || (event.target as Element).closest('button')) return;
      const rect = this.host.getBoundingClientRect();
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY, start: [...this.position] };
      heading.setPointerCapture(event.pointerId);
      heading.dataset.scaleX = String(rect.width / this.host.clientWidth);
      heading.dataset.scaleY = String(rect.height / this.host.clientHeight);
      event.preventDefault();
    };
    heading.onpointermove = (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      this.position = [
        drag.start[0] + (event.clientX - drag.x) / Number(heading.dataset.scaleX),
        drag.start[1] + (event.clientY - drag.y) / Number(heading.dataset.scaleY),
      ];
      this.place();
    };
    heading.onpointerup = heading.onpointercancel = () => {
      drag = null;
    };
    for (const joint of choices) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'viewport-joint-picker-row';
      row.textContent = `${joint.name}  (${joint.type})`;
      row.disabled = !supported(joint);
      row.onclick = () => {
        this.remembered.set(body.id, joint.id);
        this.callbacks.select(joint.id);
      };
      row.ondblclick = () => {
        this.callbacks.select(joint.id);
        this.dismiss();
        this.callbacks.focus(joint.id);
        this.callbacks.restoreFocus();
      };
      panel.append(row);
      this.rows.set(joint.id, row);
    }
    for (const event of ['pointerdown', 'pointermove', 'pointerup', 'dblclick', 'wheel'])
      panel.addEventListener(event, (event) => event.stopPropagation());
    panel.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Escape') {
        event.preventDefault();
        this.dismiss();
        this.callbacks.restoreFocus();
      }
    });
    this.host.append(panel);
    this.place();
    this.resize = new ResizeObserver(() => this.place());
    this.resize.observe(this.host);
    this.resize.observe(panel);
  }
  dispose() {
    this.remove();
    this.remembered.clear();
  }
}
