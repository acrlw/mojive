import { observeSurfaces } from './surfaces';
import './precise-input.css';
export interface PreciseGizmoTarget {
  action: string;
  label: string;
  unit: 'm' | '°' | '×';
  absoluteValue?: number;
  valid: () => boolean;
  apply: (value: number, absolute: boolean) => void;
}
/** Native precise-input semantics: edits are drafts until Enter, with mode/unit conversion. */
export class PreciseInputPopover {
  private layer: HTMLDivElement | null = null;
  private target: PreciseGizmoTarget | null = null;
  private releaseSurfaces: (() => void) | null = null;
  private resize: ResizeObserver | null = null;
  constructor(
    private host: HTMLElement,
    private onClose: () => void = () => {},
    private translate: (text: string) => string = (text) => text,
  ) {}
  get active() {
    return this.layer !== null;
  }
  validate() {
    if (this.target && !this.target.valid()) this.close();
  }
  close() {
    if (!this.layer) return;
    this.releaseSurfaces?.();
    this.releaseSurfaces = null;
    this.resize?.disconnect();
    this.resize = null;
    this.layer.remove();
    this.layer = null;
    this.target = null;
    this.onClose();
  }
  show(target: PreciseGizmoTarget, point: [number, number]) {
    this.close();
    this.target = target;
    const layer = document.createElement('div');
    layer.className = 'precise-gizmo-layer';
    this.layer = layer;
    const panel = document.createElement('form');
    panel.className = 'precise-gizmo-input';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute(
      'aria-label',
      `${this.translate(target.action)} ${this.translate(target.label)}`,
    );
    const position = () => {
      panel.style.left = `${Math.max(8, Math.min(this.host.clientWidth - panel.offsetWidth - 8, point[0] + 12))}px`;
      panel.style.top = `${Math.max(8, Math.min(this.host.clientHeight - panel.offsetHeight - 8, point[1] + 12))}px`;
    };
    const title = document.createElement('strong');
    title.textContent = `${this.translate(target.action)} ${this.translate(target.label)}`;
    panel.append(title);
    let absolute =
        localStorage.getItem('mojive.precise.absolute') === 'true' &&
        target.absoluteValue !== undefined,
      radians = localStorage.getItem('mojive.precise.radians') === 'true';
    const modes = document.createElement('div');
    modes.className = 'precise-gizmo-segments';
    const relative = document.createElement('button'),
      absoluteButton = document.createElement('button');
    relative.type = absoluteButton.type = 'button';
    relative.textContent = this.translate('Relative');
    absoluteButton.textContent = this.translate('Absolute');
    modes.append(relative, absoluteButton);
    if (target.absoluteValue !== undefined) panel.append(modes);
    const row = document.createElement('div');
    row.className = 'precise-gizmo-number';
    const input = document.createElement('input');
    input.type = 'text';
    input.inputMode = 'decimal';
    input.setAttribute('aria-label', 'Gizmo value');
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.value = String(
      absolute ? target.absoluteValue! * (target.unit === '°' && radians ? Math.PI / 180 : 1) : 0,
    );
    row.append(input);
    const unit = document.createElement('div');
    unit.className = 'precise-gizmo-segments units';
    const degrees = document.createElement('button'),
      rad = document.createElement('button');
    degrees.type = rad.type = 'button';
    degrees.textContent = 'deg';
    rad.textContent = 'rad';
    unit.append(degrees, rad);
    if (target.unit === '°') row.append(unit);
    else {
      const suffix = document.createElement('span');
      suffix.textContent = target.unit;
      row.append(suffix);
    }
    panel.append(row);
    const error = document.createElement('p');
    error.className = 'precise-gizmo-error';
    error.setAttribute('role', 'alert');
    panel.append(error);
    input.title =
      target.unit === '°'
        ? this.translate('Enter to apply · Esc to cancel · U to change unit')
        : this.translate('Enter to apply · Esc to cancel');
    const update = () => {
      relative.classList.toggle('active', !absolute);
      absoluteButton.classList.toggle('active', absolute);
      degrees.classList.toggle('active', !radians);
      rad.classList.toggle('active', radians);
      relative.setAttribute('aria-pressed', String(!absolute));
      absoluteButton.setAttribute('aria-pressed', String(absolute));
      degrees.setAttribute('aria-pressed', String(!radians));
      rad.setAttribute('aria-pressed', String(radians));
      position();
    };
    const changeMode = (next: boolean) => {
      if (absolute === next) return;
      const n = Number(input.value),
        ref = target.absoluteValue! * (target.unit === '°' && radians ? Math.PI / 180 : 1);
      if (input.value.trim() && Number.isFinite(n)) input.value = String(n + (next ? ref : -ref));
      absolute = next;
      localStorage.setItem('mojive.precise.absolute', String(next));
      update();
      input.focus();
      input.select();
    };
    const changeUnit = (next: boolean) => {
      if (radians === next) return;
      const n = Number(input.value);
      if (input.value.trim() && Number.isFinite(n))
        input.value = String(n * (next ? Math.PI / 180 : 180 / Math.PI));
      radians = next;
      localStorage.setItem('mojive.precise.radians', String(next));
      update();
      input.focus();
      input.select();
    };
    relative.onclick = () => changeMode(false);
    absoluteButton.onclick = () => changeMode(true);
    degrees.onclick = () => changeUnit(false);
    rad.onclick = () => changeUnit(true);
    panel.onsubmit = (e) => {
      e.preventDefault();
      const value = Number(input.value);
      try {
        if (!input.value.trim() || !Number.isFinite(value))
          throw new Error('Enter a finite numeric value');
        target.apply(value * (target.unit === '°' && radians ? 180 / Math.PI : 1), absolute);
        this.close();
      } catch (cause) {
        error.textContent = this.translate(cause instanceof Error ? cause.message : String(cause));
        update();
        input.focus();
      }
    };
    layer.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      if (e.target === layer) {
        e.preventDefault();
        this.close();
      }
    });
    layer.addEventListener('dblclick', (e) => e.stopPropagation());
    layer.addEventListener('wheel', (e) => e.stopPropagation());
    layer.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Escape') {
        e.preventDefault();
        this.close();
      } else if (e.key.toLowerCase() === 'u' && target.unit === '°') {
        e.preventDefault();
        changeUnit(!radians);
      }
    });
    layer.append(panel);
    this.host.append(layer);
    update();
    this.releaseSurfaces = observeSurfaces(
      [panel, modes, relative, absoluteButton, input, unit, degrees, rad].filter(
        (element) => element.isConnected,
      ),
    );
    this.resize = new ResizeObserver(position);
    this.resize.observe(panel);
    this.resize.observe(this.host);
    input.focus();
    input.select();
  }
  dispose() {
    this.close();
  }
}
