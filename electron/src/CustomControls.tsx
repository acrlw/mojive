import { useId, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { Button, Icon, Squircle } from './components';
import { t } from './i18n';

/** Popovers remain descendants of their owning dialog and enter the browser top layer. */
export function useControlPopover(
  trigger: RefObject<HTMLElement | null>,
  popup: RefObject<HTMLDivElement | null>,
  open: boolean,
  close: () => void,
) {
  useLayoutEffect(() => {
    const node = popup.current,
      anchor = trigger.current;
    if (!node || !anchor || !open) return;
    const position = () => {
      const box = anchor.getBoundingClientRect();
      let zoom = 1;
      for (let element: HTMLElement | null = node; element; element = element.parentElement)
        zoom *= parseFloat(getComputedStyle(element).zoom) || 1;
      node.style.setProperty('--popover-zoom', String(zoom));
      node.style.maxHeight = `${(innerHeight - 16) / zoom}px`;
      node.style.maxWidth = `${(innerWidth - 16) / zoom}px`;
      const width = node.offsetWidth * zoom,
        height = node.offsetHeight * zoom;
      node.style.left = `${Math.max(8, Math.min(box.left, innerWidth - width - 8)) / zoom}px`;
      node.style.top = `${Math.max(8, Math.min(box.bottom + 4, innerHeight - height - 8)) / zoom}px`;
    };
    node.showPopover();
    position();
    const resize = new ResizeObserver(position);
    resize.observe(node);
    const toggle = (event: Event) => {
      if ((event as ToggleEvent).newState === 'closed') close();
    };
    node.addEventListener('toggle', toggle);
    window.addEventListener('resize', position);
    return () => {
      resize.disconnect();
      node.removeEventListener('toggle', toggle);
      window.removeEventListener('resize', position);
      if (node.matches(':popover-open')) node.hidePopover();
    };
  }, [open]);
}
export interface SelectOption<T extends string | number> {
  value: T;
  label: string;
  icon?: string;
  disabled?: boolean;
}
export function Select<T extends string | number>({
  value,
  options,
  onChange,
  label,
  disabled = false,
  className = '',
  ...props
}: {
  value: T;
  options: readonly SelectOption<T>[];
  onChange: (value: T) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
  'aria-label'?: string;
}) {
  const trigger = useRef<HTMLDivElement>(null),
    popup = useRef<HTMLDivElement>(null),
    id = useId();
  const [open, setOpen] = useState(false),
    [highlight, setHighlight] = useState(0);
  const selected = options.find((option) => option.value === value);
  const close = () => setOpen(false);
  useControlPopover(trigger, popup, open, close);
  const show = () => {
    setHighlight(
      Math.max(
        0,
        options.findIndex((option) => option.value === value),
      ),
    );
    setOpen(true);
  };
  const choose = (index: number) => {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    close();
    trigger.current?.querySelector('button')?.focus();
  };
  return (
    <div ref={trigger} className={`custom-select ${className}`}>
      <Button
        className="select-trigger"
        role="combobox"
        aria-label={props['aria-label'] ?? (label ? t(label) : undefined)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-activedescendant={open ? `${id}-${highlight}` : undefined}
        disabled={disabled}
        onClick={() => (open ? close() : show())}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && open) {
            event.preventDefault();
            event.stopPropagation();
            close();
            return;
          }
          if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            if (!open) {
              show();
              return;
            }
            const direction = event.key === 'ArrowUp' ? -1 : 1;
            let index =
              event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? options.length - 1
                  : highlight + direction;
            index = (index + options.length) % options.length;
            for (let tries = 0; tries < options.length && options[index]?.disabled; tries++)
              index = (index + direction + options.length) % options.length;
            setHighlight(index);
            popup.current
              ?.querySelector(`#${CSS.escape(id + '-' + index)}`)
              ?.scrollIntoView({ block: 'nearest' });
          } else if ((event.key === 'Enter' || event.key === ' ') && open) {
            event.preventDefault();
            choose(highlight);
          } else if (event.key === 'Tab') close();
          else if (event.key.length === 1 && !event.metaKey && !event.ctrlKey) {
            const index = options.findIndex(
              (option) =>
                !option.disabled &&
                t(option.label).toLocaleLowerCase().startsWith(event.key.toLocaleLowerCase()),
            );
            if (index >= 0) {
              if (open) setHighlight(index);
              else onChange(options[index].value);
            }
          }
        }}
      >
        {selected?.icon && <Icon name={selected.icon} />}
        <span>{selected ? t(selected.label) : String(value)}</span>
        <Icon name="down" />
      </Button>
      {open && (
        <div ref={popup} popover="auto" className="control-popover select-popover">
          <Squircle
            radius={8}
            className="select-options"
            role="listbox"
            id={id}
            aria-label={props['aria-label'] ?? (label ? t(label) : undefined)}
          >
            {options.map((option, index) => (
              <Button
                key={option.value}
                id={`${id}-${index}`}
                role="option"
                aria-selected={value === option.value}
                disabled={option.disabled}
                className={`select-option ${highlight === index ? 'highlighted' : ''}`}
                onPointerMove={() => setHighlight(index)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => choose(index)}
              >
                {option.icon && <Icon name={option.icon} />}
                <span>{t(option.label)}</span>
                {value === option.value && <Icon name="check" />}
              </Button>
            ))}
          </Squircle>
        </div>
      )}
    </div>
  );
}
export function Switch({
  checked,
  onChange,
  label,
  disabled = false,
  className = '',
  ...props
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
  'aria-label'?: string;
  id?: string;
}) {
  return (
    <Button
      {...props}
      type="button"
      className={`custom-switch ${className}`}
      data-round-surface=""
      role="switch"
      aria-label={props['aria-label'] ?? (label ? t(label) : undefined)}
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span data-round-surface="" />
    </Button>
  );
}

/** The labeled row is the sole pointer, keyboard and accessibility owner. */
export function SwitchRow({
  checked,
  onChange,
  label,
  disabled = false,
  className = '',
  title,
  ...props
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  className?: string;
  'aria-label'?: string;
  title?: string;
  id?: string;
}) {
  return (
    <button
      {...props}
      type="button"
      className={`button switch-row ${className}`}
      role="switch"
      aria-label={props['aria-label'] ? t(props['aria-label']) : undefined}
      aria-checked={checked}
      title={title ? t(title) : undefined}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className="switch-row-label">{t(label)}</span>
      <span className="custom-switch" data-round-surface="" aria-hidden="true">
        <span data-round-surface="" />
      </span>
    </button>
  );
}
