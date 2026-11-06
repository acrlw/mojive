import { Fragment, useRef, useState } from 'react';
import catalog from '../main/menu-catalog.json';
import en from '../main/menu-en.json';
import zh from '../main/menu-zh.json';
import { AppMenu } from './AppMenu';
import { Icon } from './components';
import { isMacPlatform } from './platform';
import type { NativeMenuState } from './contracts';
import './workspace-menu.css';

interface MenuEntry {
  type?: 'separator' | 'checkbox' | 'radio';
  action?: string;
  label?: string;
  icon?: string;
  activeLabel?: string;
  activeIcon?: string;
  enabled?: string;
  checked?: { key: string; value: string | number | boolean };
  accelerator?: string;
  shortcut?: string;
  children?: MenuEntry[];
}
type MenuState = NativeMenuState & { blocked?: boolean };
const groups = catalog as { name: string; label: string; items: MenuEntry[] }[];
const labels: Record<string, Record<string, string>> = { en, 'zh-CN': zh };
export const workspaceMenus = groups.map(({ name, label }) => ({ name, label }));
export function workspaceMenuLabel(key: string, locale: string) {
  return labels[locale]?.[key] ?? labels.en[key] ?? key;
}
function acceleratorLabel(value: unknown) {
  if (typeof value !== 'string') return '';
  const mac = isMacPlatform();
  const keys = value.split('+');
  if (mac) {
    const modifiers = ['Control', 'Alt', 'Shift', 'Meta', 'Command', 'CommandOrControl'];
    keys.sort((a, b) => {
      const rank = (key: string) => modifiers.includes(key) ? modifiers.indexOf(key) : modifiers.length;
      return rank(a) - rank(b);
    });
  }
  return keys.map((key) => {
    if (key === 'CommandOrControl') return mac ? '⌘' : 'Ctrl';
    if (key === 'Meta' || key === 'Command') return mac ? '⌘' : 'Super';
    if (key === 'Control') return mac ? '⌃' : 'Ctrl';
    if (key === 'Shift') return mac ? '⇧' : 'Shift';
    if (key === 'Alt') return mac ? '⌥' : 'Alt';
    if (key === 'Backspace') return mac ? '⌫' : 'Backspace';
    if (key === 'ArrowLeft') return '←';
    if (key === 'ArrowRight') return '→';
    return key;
  }).join(mac ? ' ' : '+');
}

function MenuItems({ entries, state, onAction, onClose }: {
  entries: MenuEntry[];
  state: MenuState;
  onAction: (action: string) => void;
  onClose: () => void;
}) {
  const [submenu, setSubmenu] = useState<number | null>(null);
  const anchors = useRef(new Map<number, HTMLButtonElement>());
  const values = state as unknown as Record<string, unknown>;
  const label = (key: string) => workspaceMenuLabel(key, state.locale);
  return entries.map((entry, index) => {
    if (entry.type === 'separator') return <hr key={index} role="separator" />;
    const checked = Boolean(entry.checked && values[entry.checked.key] === entry.checked.value);
    const enabled = !state.blocked && (!entry.enabled || Boolean(values[entry.enabled]));
    const icon = checked && entry.activeIcon ? entry.activeIcon : entry.icon;
    const title = label((checked && entry.activeLabel ? entry.activeLabel : entry.label) ?? '');
    const accelerator = acceleratorLabel(entry.shortcut ? values[entry.shortcut] : entry.accelerator);
    const anchor = anchors.current.get(index);
    const rect = anchor?.getBoundingClientRect();
    const openSubmenu = () => { if (enabled && entry.children) setSubmenu(index); };
    return <Fragment key={entry.action ?? entry.label}>
      <button
        ref={(node) => { if (node) anchors.current.set(index, node); else anchors.current.delete(index); }}
        className="workspace-menu-row"
        role={entry.type === 'checkbox' ? 'menuitemcheckbox' : entry.type === 'radio' ? 'menuitemradio' : 'menuitem'}
        aria-checked={entry.type === 'checkbox' || entry.type === 'radio' ? checked : undefined}
        aria-haspopup={entry.children ? 'menu' : undefined}
        aria-expanded={entry.children ? submenu === index : undefined}
        disabled={!enabled}
        onPointerEnter={() => entry.children && enabled ? setSubmenu(index) : setSubmenu(null)}
        onFocus={() => { if (submenu !== null && submenu !== index) setSubmenu(null); }}
        onKeyDown={(event) => {
          if (entry.children && event.key === 'ArrowRight') {
            event.preventDefault(); event.stopPropagation(); openSubmenu();
          }
        }}
        onClick={() => {
          if (entry.children) openSubmenu();
          else if (entry.action) { onAction(entry.action); onClose(); }
        }}
      >
        <span className="workspace-menu-icon">{icon && <Icon name={icon} size={16} />}</span>
        <span className="workspace-menu-label">{title}</span>
        {accelerator && <kbd>{accelerator}</kbd>}
        <span className="workspace-menu-indicator">
          {entry.children ? <Icon name="chevron-right" size={14} /> : checked && entry.type ? <Icon name="check" size={14} /> : null}
        </span>
      </button>
      {entry.children && submenu === index && anchor && rect && <AppMenu
        name={`submenu-${entry.label}`}
        anchor={anchor}
        point={{ x: rect.right + 4, y: rect.top - 4 }}
        onClose={() => setSubmenu(null)}
        onNavigate={(direction) => {
          if (direction < 0) { setSubmenu(null); anchor.focus({ preventScroll: true }); }
        }}
      ><MenuItems entries={entry.children} state={state} onAction={onAction} onClose={onClose} /></AppMenu>}
    </Fragment>;
  });
}

export function WorkspaceMenuItems({ name, state, onAction, onClose }: {
  name: string;
  state: MenuState;
  onAction: (action: string) => void;
  onClose: () => void;
}) {
  const entries = groups.find((group) => group.name === name)?.items ?? [];
  return <MenuItems entries={entries} state={state} onAction={onAction} onClose={onClose} />;
}
