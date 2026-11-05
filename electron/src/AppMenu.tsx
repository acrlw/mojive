import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { useControlPopover } from './CustomControls';

/** Menu paint, shadow, placement and keyboard navigation share one popup owner. */
export function AppMenu({
  name,
  onClose,
  anchor,
  point,
  onNavigate,
  children,
}: {
  name: string;
  onClose: () => void;
  anchor?: HTMLElement | null;
  point?: { x: number; y: number };
  onNavigate?: (direction: number) => void;
  children: ReactNode;
}) {
  const popup = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLElement | null>(
    anchor ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null),
  );
  useControlPopover(trigger, popup, true, onClose, point);
  useLayoutEffect(() => {
    popup.current
      ?.querySelector<HTMLButtonElement>('button:not(:disabled)')
      ?.focus({ preventScroll: true });
  }, []);
  useLayoutEffect(() => {
    if (!point) return;
    // Electron on Linux emits contextmenu before pointerup; auto popovers would
    // light-dismiss the new menu on that same right-button release.
    const dismiss = (event: PointerEvent) => {
      if (!popup.current?.contains(event.target as Node)) onClose();
    };
    document.addEventListener('pointerdown', dismiss);
    window.addEventListener('blur', onClose);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      window.removeEventListener('blur', onClose);
    };
  }, [point]);
  return (
    <div
      ref={popup}
      popover={point ? 'manual' : 'auto'}
      className={`menu menu-${name.toLowerCase()}`}
      role="menu"
      onKeyDown={(event) => {
        event.stopPropagation();
        if (onNavigate && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
          event.preventDefault();
          onNavigate(event.key === 'ArrowLeft' ? -1 : 1);
          return;
        }
        const items = [
          ...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'),
        ];
        if (event.key === 'Escape' || event.key === 'Tab') {
          event.preventDefault();
          onClose();
          trigger.current?.focus({ preventScroll: true });
          return;
        }
        const keys = ['ArrowDown', 'ArrowUp', 'Home', 'End'];
        if (keys.includes(event.key)) {
          event.preventDefault();
          const index = items.indexOf(document.activeElement as HTMLButtonElement);
          const next =
            event.key === 'Home'
              ? 0
              : event.key === 'End'
                ? items.length - 1
                : (index + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
          items[next]?.focus({ preventScroll: true });
        }
      }}
    >
      {children}
    </div>
  );
}
