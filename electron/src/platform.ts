/** Desktop preload is authoritative; browser previews use the browser's host platform. */
export function isMacPlatform() {
  return window.mojive?.platform
    ? window.mojive.platform === 'darwin'
    : /Mac/i.test(navigator.platform);
}

export function systemKeyLabel() {
  const platform = window.mojive?.platform ?? navigator.platform;
  return isMacPlatform() ? '⌘' : /Win/i.test(platform) ? 'Win' : 'Super';
}

export function commandKey(event: { metaKey: boolean; ctrlKey: boolean }) {
  return isMacPlatform() ? event.metaKey : event.ctrlKey;
}

export function shortcutParts(key: string, modifiers: ('command' | 'shift' | 'alt')[] = []) {
  const mac = isMacPlatform();
  const names = { command: mac ? '⌘' : 'Ctrl', shift: mac ? '⇧' : 'Shift', alt: mac ? '⌥' : 'Alt' };
  const order: (keyof typeof names)[] = mac
    ? ['shift', 'alt', 'command']
    : ['command', 'alt', 'shift'];
  return [
    ...order.filter((modifier) => modifiers.includes(modifier)).map((modifier) => names[modifier]),
    key === 'Delete' && mac ? '⌫' : key,
  ];
}

export function shortcut(key: string, modifiers: ('command' | 'shift' | 'alt')[] = []) {
  return shortcutParts(key, modifiers).join(isMacPlatform() ? ' ' : '+');
}
