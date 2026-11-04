import { useId, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from 'react';
import { Icon, Squircle } from './components';
import './entity-card.css';

type Props = HTMLAttributes<HTMLDivElement> & {
  name: string;
  icon: string;
  context?: ReactNode;
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
  selected,
  expanded,
  heading,
  contentId,
  disclosureTitle,
  className = '',
  children,
  ...props
}: Props) {
  const id = useId();
  const bodyId = contentId ?? `${id}-content`;
  return (
    <Squircle
      radius={12}
      className={`entity-card ${selected ? 'selected' : ''} ${className}`}
      {...props}
    >
      <button
        type="button"
        className="entity-card-heading"
        aria-expanded={expanded}
        aria-controls={expanded === undefined ? undefined : bodyId}
        {...heading}
      >
        <Icon name={icon} size={20} />
        <span className="entity-card-title">{name}</span>
        {context !== undefined && <span className="entity-card-context">{context}</span>}
        {expanded !== undefined && (
          <span className="entity-card-disclosure" aria-hidden="true" title={disclosureTitle}>
            <Icon name={expanded ? 'down' : 'right'} size={20} />
          </span>
        )}
      </button>
      <div className="entity-card-content" id={bodyId}>
        {children}
      </div>
    </Squircle>
  );
}
