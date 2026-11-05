import { isMacPlatform, systemKeyLabel } from './platform';

export const FLY_ACTIONS = {
  flyForward: 'Fly forward',
  flyBack: 'Fly backward',
  flyLeft: 'Fly left',
  flyRight: 'Fly right',
  flyUp: 'Fly up',
  flyDown: 'Fly down',
} as const;
export const INPUT_ACTIONS = {
  select: 'Select tool',
  position: 'Position tool',
  rotation: 'Rotation tool',
  scale: 'Scale tool',
  joint: 'Joint tool',
  focus: 'Focus',
  frame: 'World / Body',
  snap: 'Snapping',
  capture: 'Capture keyframe',
  playback: 'Playback',
  ...FLY_ACTIONS,
} as const;
export type InputAction = keyof typeof INPUT_ACTIONS;
export type NavigationAction = 'orbit' | 'pan' | 'dolly' | 'zoom';
export const NAVIGATION_ACTIONS: Record<NavigationAction, string> = {
  orbit: 'Orbit',
  pan: 'Pan',
  dolly: 'Dolly drag',
  zoom: 'Zoom',
};
export interface InputBindings {
  keys: Record<InputAction, string | null>;
  pointer: Record<NavigationAction, string[]>;
}
export const KEY_CHOICES = [
  'Space',
  'Shift',
  'Control',
  'Alt',
  'Meta',
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
  ...Array.from({ length: 12 }, (_, i) => `F${i + 1}`),
];
export const DEFAULT_INPUT_BINDINGS: InputBindings = {
  keys: {
    select: 'V',
    position: 'G',
    rotation: 'R',
    scale: null,
    joint: 'J',
    focus: 'F',
    frame: 'T',
    snap: 'Shift',
    capture: 'K',
    playback: 'Space',
    flyForward: 'W',
    flyBack: 'S',
    flyLeft: 'A',
    flyRight: 'D',
    flyUp: 'Q',
    flyDown: 'E',
  },
  pointer: {
    orbit: ['left'],
    pan: ['right', 'middle', 'snap+left', 'snap+right', 'snap+middle'],
    dolly: [],
    zoom: ['wheel'],
  },
};
export const NAVIGATION_PRESETS = {
  Mojive: DEFAULT_INPUT_BINDINGS.pointer,
  Blender: { orbit: ['middle'], pan: ['shift+middle'], dolly: ['ctrl+middle'], zoom: ['wheel'] },
  Unity: { orbit: ['alt+left'], pan: ['middle'], dolly: ['alt+right'], zoom: ['wheel'] },
  Unreal: { orbit: ['alt+left'], pan: ['alt+middle'], dolly: ['alt+right'], zoom: ['wheel'] },
  MuJoCo: {
    orbit: ['left', 'shift+left'],
    pan: ['right', 'shift+right'],
    dolly: ['middle'],
    zoom: ['wheel'],
  },
} satisfies Record<string, InputBindings['pointer']>;
export function keyLabel(key: string | null) {
  if (!key) return '';
  const mac = isMacPlatform();
  return (
    (
      {
        Shift: mac ? '⇧' : 'Shift',
        Control: mac ? '⌃' : 'Ctrl',
        Alt: mac ? '⌥' : 'Alt',
        Meta: systemKeyLabel(),
      } as Record<string, string>
    )[key] ?? key
  );
}
export function actionParts(bindings: InputBindings, action: InputAction) {
  return bindings.keys[action]?.split('+').map(keyLabel) ?? [];
}
export function actionLabel(bindings: InputBindings, action: InputAction) {
  return actionParts(bindings, action).join(isMacPlatform() ? ' ' : '+');
}
export function eventKey(event: { key: string }) {
  return event.key === ' ' ? 'Space' : event.key.length === 1 ? event.key.toUpperCase() : event.key;
}
export type Modifiers = { shiftKey: boolean; ctrlKey: boolean; altKey: boolean; metaKey: boolean };
const modifierKeys = ['Control', 'Alt', 'Shift', 'Meta'];
export function keyboardChord(event: Modifiers & { key: string }) {
  const key = eventKey(event);
  const down = [event.ctrlKey, event.altKey, event.shiftKey, event.metaKey];
  return [...modifierKeys.filter((name, index) => down[index] && name !== key), key].join('+');
}
function validKeyChord(chord: string) {
  const parts = chord.split('+'),
    key = parts.at(-1)!;
  if (
    !KEY_CHOICES.includes(key) ||
    parts.slice(0, -1).some((part) => !modifierKeys.includes(part)) ||
    new Set(parts).size !== parts.length
  )
    return false;
  // Standard editing/file commands retain their native input owner.
  return (
    !/^(?:Control|Meta)\+(?:(?:Control|Alt|Shift|Meta)\+)*[OSZACVD]$/.test(chord) &&
    chord !== 'Shift+F10' &&
    chord !== 'Alt+F4' &&
    // These roles are installed by the macOS application/window menu.
    !(
      isMacPlatform() &&
      [
        'Meta+Q',
        'Meta+W',
        'Meta+H',
        'Alt+Meta+H',
        'Meta+M',
        'Control+Meta+F',
        'Alt+Meta+I',
      ].includes(chord)
    )
  );
}
export function matchesKey(
  bindings: InputBindings,
  action: InputAction,
  event: Modifiers & { key: string },
) {
  return bindings.keys[action] === keyboardChord(event);
}
export function focusAccelerator(bindings: InputBindings) {
  const key = bindings.keys.focus;
  return key && !modifierKeys.includes(key.split('+').at(-1)!) ? key : null;
}
export function focusAllAccelerator(bindings: InputBindings) {
  const key = focusAccelerator(bindings);
  if (!key || key.split('+').includes('Shift')) return null;
  const parts = key.split('+'),
    primary = parts.pop()!;
  const chord = [
    ...modifierKeys.filter((modifier) => modifier === 'Shift' || parts.includes(modifier)),
    primary,
  ].join('+');
  return Object.values(bindings.keys).includes(chord) ? null : chord;
}
export function keyHeld(key: string | null, event: Modifiers, held: ReadonlySet<string>): boolean {
  if (!key) return false;
  return key
    .split('+')
    .every((part) =>
      part === 'Shift'
        ? event.shiftKey
        : part === 'Control'
          ? event.ctrlKey
          : part === 'Alt'
            ? event.altKey
            : part === 'Meta'
              ? event.metaKey
              : held.has(part),
    );
}
export function mouseChord(event: Modifiers & { button?: number }, wheel = false) {
  return [
    ...(event.ctrlKey ? ['ctrl'] : []),
    ...(event.altKey ? ['alt'] : []),
    ...(event.shiftKey ? ['shift'] : []),
    ...(event.metaKey ? ['meta'] : []),
    wheel ? 'wheel' : ['left', 'middle', 'right'][event.button ?? -1],
  ].join('+');
}
const modifiers: Record<string, string> = {
  shift: 'Shift',
  ctrl: 'Control',
  alt: 'Alt',
  meta: 'Meta',
};
export function pointerParts(chord: string, bindings: InputBindings) {
  const parts = chord.split('+');
  return {
    button: parts.at(-1)!,
    keys: parts
      .slice(0, -1)
      .flatMap((key) =>
        key === 'snap' ? (bindings.keys.snap?.split('+') ?? [null]) : [modifiers[key]],
      ),
  };
}
export function matchesPointer(
  bindings: InputBindings,
  action: NavigationAction,
  event: Modifiers & { button?: number },
  held: ReadonlySet<string>,
  wheel = false,
) {
  const button = wheel ? 'wheel' : ['left', 'middle', 'right'][event.button ?? -1];
  return bindings.pointer[action].some((chord) => {
    const parsed = pointerParts(chord, bindings);
    if (parsed.button !== button || parsed.keys.some((key) => !key || !keyHeld(key, event, held)))
      return false;
    return Object.values(modifiers).every((key) => {
      const pressed = key === 'Meta' ? event.metaKey : keyHeld(key, event, held);
      return !pressed || parsed.keys.includes(key);
    });
  });
}
function validatePointers(bindings: InputBindings) {
  const owners = new Map<string, string>();
  for (const action of Object.keys(NAVIGATION_ACTIONS) as NavigationAction[])
    for (const chord of bindings.pointer[action]) {
      const parts = chord.split('+'),
        button = parts.at(-1);
      const allowed = action === 'zoom' ? ['wheel'] : ['left', 'middle', 'right'];
      if (
        !allowed.includes(button!) ||
        parts.slice(0, -1).some((part) => part !== 'snap' && !modifiers[part]) ||
        new Set(parts).size !== parts.length
      )
        throw new Error('Use the mouse wheel for zoom and a mouse button for dragging.');
      const resolved = pointerParts(chord, bindings);
      if (resolved.keys.includes(null)) continue;
      const identity = [...new Set(resolved.keys)].sort().join('+') + '+' + button;
      if (owners.has(identity) && owners.get(identity) !== action)
        throw new Error('A mouse gesture is already assigned to another navigation action.');
      owners.set(identity, action);
    }
}
/** Like the native editor, assigning an occupied key swaps the two actions atomically. */
export function remapKey(bindings: InputBindings, action: InputAction, key: string | null) {
  if (key !== null && !validKeyChord(key)) throw new Error('Unsupported shortcut');
  const keys = { ...bindings.keys };
  const displaced =
    key &&
    (Object.keys(keys) as InputAction[]).find(
      (candidate) => candidate !== action && keys[candidate] === key,
    );
  if (displaced) keys[displaced] = keys[action];
  keys[action] = key;
  const result = { ...bindings, keys };
  validatePointers(result);
  return result;
}
export function remapPointer(bindings: InputBindings, action: NavigationAction, text: string) {
  const chords = text
    .split(';')
    .map((value) => value.trim().toLowerCase().replace(/\s/g, ''))
    .filter(Boolean);
  const result = { ...bindings, pointer: { ...bindings.pointer, [action]: chords } };
  validatePointers(result);
  return result;
}
export function normalizeInputBindings(value: unknown): InputBindings {
  const source = value && typeof value === 'object' ? (value as Partial<InputBindings>) : {};
  let result = {
    keys: { ...DEFAULT_INPUT_BINDINGS.keys },
    pointer: { ...DEFAULT_INPUT_BINDINGS.pointer },
  };
  let saved = source.keys;
  if (
    saved &&
    typeof saved === 'object' &&
    !Object.keys(FLY_ACTIONS).some((action) => action in saved!)
  ) {
    saved = { ...saved };
    // Migrate the former Q/W/E/R factory map; retain explicit custom bindings.
    if (
      saved.select === 'Q' &&
      saved.position === 'W' &&
      saved.rotation === 'E' &&
      saved.scale === 'R'
    ) {
      for (const action of ['select', 'position', 'rotation', 'scale'] as const)
        saved[action] = DEFAULT_INPUT_BINDINGS.keys[action];
    }
    for (const action of Object.keys(FLY_ACTIONS) as (keyof typeof FLY_ACTIONS)[])
      if (Object.values(saved).includes(result.keys[action])) result.keys[action] = null;
  }
  if (saved && typeof saved === 'object')
    for (const action of Object.keys(INPUT_ACTIONS) as InputAction[]) {
      const key: string | null | undefined = saved[action];
      if (key === null || (typeof key === 'string' && validKeyChord(key))) {
        // Load the whole map before validating pointer aliases against its final snap key.
        const keys = { ...result.keys };
        const displaced =
          key &&
          (Object.keys(keys) as InputAction[]).find(
            (other) => other !== action && keys[other] === key,
          );
        if (displaced) keys[displaced] = keys[action];
        keys[action] = key;
        result = { ...result, keys };
      }
    }
  if (source.pointer && typeof source.pointer === 'object') {
    const pointer = { ...result.pointer };
    for (const action of Object.keys(NAVIGATION_ACTIONS) as NavigationAction[]) {
      const chords = source.pointer[action];
      if (Array.isArray(chords) && chords.every((chord) => typeof chord === 'string'))
        pointer[action] = [...chords];
    }
    try {
      validatePointers({ ...result, pointer });
      result = { ...result, pointer };
    } catch {
      /* Retain valid navigation defaults for malformed preferences. */
    }
  }
  return result;
}
