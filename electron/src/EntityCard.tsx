import { useId, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from 'react';
import { Icon, Squircle } from './components';
import './entity-card.css';

type Props = HTMLAttributes<HTMLDivElement> & {
  name: string;
  icon: string;
  context?: ReactNode;
  controls?: ReactNode;
  selected?: boolean;
  expanded?: boolean;
  heading: ButtonHTMLAttributes<HTMLButtonElement>;
  contentId?: string;
  disclosureTitle?: string;
};

/** Joint and actuator cards share geometry; their callers own selection and disclosure. */
export function EntityCard({
  name,
  icon,
  context,
  controls,
  selected,
  expanded,
  heading,
  contentId,
  disclosureTitle,
  className = '',
  children,
  ...props
}: Props) {
  const { title, ...headingProps } = heading;
  const id = useId();
  const bodyId = contentId ?? `${id}-content`;
  return (
    <Squircle
      radius={12}
      className={`entity-card ${selected ? 'selected' : ''} ${className}`}
      {...props}
    >
      <div className="entity-card-header">
        <button
          type="button"
          className="entity-card-heading"
          aria-expanded={expanded}
          aria-controls={expanded === undefined ? undefined : bodyId}
          {...headingProps}
          data-tooltip={title ?? name}
          data-tooltip-overflow=".entity-card-title"
        >
          <Icon name={icon} size={20} />
          <span className="entity-card-label">
            <span className="entity-card-title">{name}</span>
          </span>
          {context !== undefined && <span className="entity-card-context">{context}</span>}
        </button>
        {controls && <div className="entity-card-controls">{controls}</div>}
        {expanded !== undefined && (
          <button
            type="button"
            className="entity-card-disclosure"
            {...headingProps}
            aria-expanded={expanded}
            aria-controls={bodyId}
            aria-label={disclosureTitle}
            data-tooltip={disclosureTitle}
          >
            <Icon name={expanded ? 'down' : 'right'} size={20} />
          </button>
        )}
      </div>
      <div className="entity-card-content" id={bodyId}>
        {children}
      </div>
    </Squircle>
  );
}
