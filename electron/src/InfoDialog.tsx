import { useLayoutEffect, useRef, type ReactNode } from 'react';

/** Renderer-owned modal: background controls become inert and retain their document state. */
export function InfoDialog({
  label,
  onClose,
  children,
  className = '',
}: {
  label: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useLayoutEffect(() => {
    const element = dialog.current!;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    let zoom = 1;
    for (let parent: HTMLElement | null = element; parent; parent = parent.parentElement)
      zoom *= parseFloat(getComputedStyle(parent).zoom) || 1;
    element.style.setProperty('--dialog-zoom', String(zoom));
    element.showModal();
    element.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
    return () => {
      element.close();
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className={`help-dialog info-dialog ${className}`}
      aria-label={label}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onKeyDown={(event) => event.stopPropagation()}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const r = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < r.left ||
          event.clientX > r.right ||
          event.clientY < r.top ||
          event.clientY > r.bottom
        )
          onClose();
      }}
    >
      {children}
    </dialog>
  );
}
