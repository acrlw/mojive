import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { useControlPopover } from './CustomControls';

/** Menu paint, shadow, placement and keyboard navigation share one popup owner. */
export function AppMenu({
  name,
  onClose,
  children,
}: {
  name: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const popup = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLElement | null>(
    document.activeElement instanceof HTMLElement ? document.activeElement : null,
  );
  useControlPopover(trigger, popup, true, onClose);
  useLayoutEffect(() => {
    popup.current
      ?.querySelector<HTMLButtonElement>('button:not(:disabled)')
      ?.focus({ preventScroll: true });
  }, []);
  return (
    <div
      ref={popup}
      popover="auto"
      className={`menu menu-${name.toLowerCase()}`}
      role="menu"
      onKeyDown={(event) => {
        event.stopPropagation();
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
