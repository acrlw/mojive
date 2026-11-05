import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useControlPopover } from './CustomControls';
import { KeyHint } from './InputHints';
import './tooltip.css';

type Hint = { owner: HTMLElement; point?: { x: number; y: number } };

function addsInformation(owner: HTMLElement) {
  const caption = owner.dataset.tooltipCaption ?? owner.dataset.tooltip;
  if (!caption) return false;
  if (owner.dataset.tooltipKeys || owner.dataset.tooltipCaption) return true;
  const overflow = owner.dataset.tooltipOverflow;
  if (overflow !== undefined) {
    const label = overflow ? owner.querySelector<HTMLElement>(overflow) : owner;
    return (
      !!label &&
      (label.scrollWidth > label.clientWidth + 1 || label.scrollHeight > label.clientHeight + 1)
    );
  }
  if (owner.querySelector('[data-icon="close"], [data-icon="search"]') && !owner.innerText.trim())
    return false;
  const text = owner.innerText.trim().replace(/\s+/g, ' ');
  if (text !== caption.trim().replace(/\s+/g, ' ')) return true;
  return [owner, ...owner.querySelectorAll<HTMLElement>('*')].some(
    (node) =>
      node.innerText?.trim() === caption.trim() &&
      (node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1),
  );
}

function Tooltip({ hint, close }: { hint: Hint; close: () => void }) {
  const trigger = useRef<HTMLElement>(hint.owner);
  const popup = useRef<HTMLDivElement>(null);
  const id = useId();
  const caption = hint.owner.dataset.tooltipCaption ?? hint.owner.dataset.tooltip;
  const keys: string[] = JSON.parse(hint.owner.dataset.tooltipKeys ?? '[]');
  useControlPopover(trigger, popup, true, close, hint.point);
  useLayoutEffect(() => {
    const owner = hint.owner;
    const previous = owner.getAttribute('aria-describedby');
    owner.setAttribute('aria-describedby', [previous, id].filter(Boolean).join(' '));
    return () => {
      if (previous == null) owner.removeAttribute('aria-describedby');
      else owner.setAttribute('aria-describedby', previous);
    };
  }, [hint.owner, id]);
  useEffect(() => {
    const resize = new ResizeObserver(() => {
      if (!addsInformation(hint.owner)) close();
    });
    resize.observe(hint.owner);
    const label = hint.owner.dataset.tooltipOverflow
      ? hint.owner.querySelector(hint.owner.dataset.tooltipOverflow)
      : null;
    if (label) resize.observe(label);
    return () => resize.disconnect();
  }, [hint.owner, close]);
  useEffect(() => {
    const observer = new MutationObserver(() => {
      if (
        !hint.owner.isConnected ||
        (hint.owner.dataset.tooltipCaption ?? hint.owner.dataset.tooltip) !== caption
      )
        close();
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-tooltip', 'data-tooltip-caption'],
    });
    return () => observer.disconnect();
  }, [hint, caption, close]);
  return (
    <div
      ref={popup}
      id={id}
      role="tooltip"
      popover="manual"
      className={`mojive-tooltip ${keys.length ? 'has-shortcut' : ''}`}
    >
      <span>{caption}</span>
      {keys.length > 0 && <KeyHint keys={keys} />}
    </div>
  );
}

/** One delayed tooltip owner replaces browser title bubbles throughout the editor. */
export function TooltipHost() {
  const [hint, setHint] = useState<Hint | null>(null);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let owner: HTMLElement | null = null;
    const hide = () => {
      clearTimeout(timer);
      owner = null;
      setHint(null);
    };
    const schedule = (event: PointerEvent | FocusEvent) => {
      const target =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-tooltip]')
          : null;
      if (target === owner) return;
      hide();
      if (
        !target ||
        !addsInformation(target) ||
        ('pointerType' in event && event.pointerType === 'touch')
      )
        return;
      if (event.type === 'focusin' && !target.matches(':focus-visible')) return;
      owner = target;
      const point =
        'clientX' in event ? { x: event.clientX + 12, y: event.clientY + 16 } : undefined;
      timer = setTimeout(() => {
        if (target.isConnected && addsInformation(target)) setHint({ owner: target, point });
      }, 450);
    };
    const leave = (event: PointerEvent | FocusEvent) => {
      if (owner && !(event.relatedTarget instanceof Node && owner.contains(event.relatedTarget)))
        hide();
    };
    document.addEventListener('pointerover', schedule);
    document.addEventListener('focusin', schedule);
    document.addEventListener('pointerout', leave);
    document.addEventListener('focusout', leave);
    document.addEventListener('pointerdown', hide, true);
    document.addEventListener('keydown', hide, true);
    document.addEventListener('scroll', hide, true);
    document.addEventListener('wheel', hide, true);
    window.addEventListener('blur', hide);
    return () => {
      hide();
      document.removeEventListener('pointerover', schedule);
      document.removeEventListener('focusin', schedule);
      document.removeEventListener('pointerout', leave);
      document.removeEventListener('focusout', leave);
      document.removeEventListener('pointerdown', hide, true);
      document.removeEventListener('keydown', hide, true);
      document.removeEventListener('scroll', hide, true);
      document.removeEventListener('wheel', hide, true);
      window.removeEventListener('blur', hide);
    };
  }, []);
  if (!hint) return null;
  // Keep inherited theme/scale and modal input ownership while entering the top layer.
  const parent = hint.owner.closest('dialog, .app-shell') ?? document.body;
  return createPortal(<Tooltip hint={hint} close={() => setHint(null)} />, parent);
}
