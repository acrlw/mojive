import { useLayoutEffect, useRef, useSyncExternalStore, type FocusEvent } from 'react';

type Mode = 'adjust' | 'adjust-copy' | 'copy' | 'edit' | null;
type Owner = { mode: Mode };
let hovered: Owner | null = null;
let focused: Owner | null = null;
let active: Owner | null = null;
let feedback = '';
let snapshot = { mode: null as Mode, feedback };
let timer: ReturnType<typeof setTimeout> | undefined;
const listeners = new Set<() => void>();

function publish() {
  const next = focused?.mode === 'edit' ? focused : (hovered ?? focused);
  if (active !== next) {
    active = next;
    feedback = '';
    clearTimeout(timer);
  }
  const mode = active?.mode ?? null;
  if (mode === snapshot.mode && feedback === snapshot.feedback) return;
  snapshot = { mode, feedback };
  listeners.forEach((listener) => listener());
}

/** Only the status bar subscribes; hovering a readout does not rerender the scene. */
export function useNumericStatus() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    () => snapshot,
  );
}

export function numericCopyFeedback() {
  const owner = active;
  return (message: string) => {
    if (!owner || active !== owner) return;
    feedback = message;
    clearTimeout(timer);
    timer = setTimeout(() => {
      feedback = '';
      publish();
    }, 2000);
    publish();
  };
}

export function useNumericHint(mode: Mode) {
  const owner = useRef<Owner>({ mode });
  useLayoutEffect(() => {
    owner.current.mode = mode;
    publish();
  }, [mode]);
  useLayoutEffect(
    () => () => {
      if (hovered === owner.current) hovered = null;
      if (focused === owner.current) focused = null;
      publish();
    },
    [],
  );
  return {
    onPointerEnter: () => {
      hovered = owner.current;
      publish();
    },
    onPointerLeave: () => {
      if (hovered === owner.current) hovered = null;
      publish();
    },
    onFocusCapture: (event: FocusEvent<HTMLElement>) => {
      // Pointer scrubbing focuses the wrapper for shortcuts, not text input.
      if (event.target instanceof HTMLInputElement || event.target.matches(':focus-visible')) {
        focused = owner.current;
        publish();
      }
    },
    onBlurCapture: (event: FocusEvent<HTMLElement>) => {
      if (event.currentTarget.contains(event.relatedTarget)) return;
      if (focused === owner.current) focused = null;
      publish();
    },
  };
}
