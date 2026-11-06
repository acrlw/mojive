import { Button, Numeric, Section, Select, SwitchRow, VectorFields } from '../components';
import { t } from '../i18n';
import type { EnvironmentField, EnvironmentValue, EnvironmentValues } from './contracts';
import './simulation.css';

interface Props {
  fields: EnvironmentField[];
  values: EnvironmentValues;
  onBegin: () => void;
  onCommit: () => void;
  onCancel: () => void;
  onChange: (values: EnvironmentValues) => void;
  onReset: () => void;
}

/** The adapter declares available options; the presenter only chooses shared controls. */
export function EnvironmentProperties(props: Props) {
  const { fields, values, onBegin, onCommit, onCancel, onChange, onReset } = props;
  const input = { onBegin, onCommit, onCancel };
  const edit = (id: string, value: EnvironmentValue) => {
    onBegin(); onChange({ [id]: value }); onCommit();
  };
  const renderField = (field: EnvironmentField) => {
    const value = values[field.id], label = t(field.label);
    if (field.type === 'boolean') return <SwitchRow key={field.id} label={label}
      checked={value as boolean} onChange={(checked) => edit(field.id, checked)} />;
    if (field.type === 'vector') {
      const numbers = value as number[];
      const change = (index: number, next: number) => {
        const updated = [...numbers]; updated[index] = next; onChange({ [field.id]: updated });
      };
      if (field.size === 3) return <VectorFields key={field.id} label={label} values={numbers}
        unit={field.unit} step={field.step} {...input} onChange={change} />;
      return <div key={field.id} className="environment-vector">
        <label>{label}</label>
        <div>{numbers.map((number, index) => <Numeric key={index} label={`${label} ${index + 1}`}
          value={number} step={field.step} min={field.min} max={field.max} {...input}
          onChange={(next) => change(index, next)} />)}</div>
      </div>;
    }
    return <div className="property-row environment-property" key={field.id}>
      <label>{label}{field.type === 'number' && field.unit && <span> ({field.unit})</span>}</label>
      {field.type === 'enum'
        ? <Select label={label} value={value as number}
            options={field.choices.map((choice) => ({ ...choice, label: t(choice.label) }))}
            onChange={(next) => edit(field.id, next)} />
        : <Numeric label={label} value={value as number} {...input} min={field.min} max={field.max}
            integer={field.integer} step={field.step} precision={field.integer ? 0 : 6}
            formatValue={(number) => Number(number.toPrecision(7)).toString()}
            onChange={(next) => onChange({ [field.id]: next })} />}
    </div>;
  };
  const groups = [...new Set(fields.filter((field) => field.advanced).map((field) => field.group))];
  return <div className="environment-properties">
    <Section title={t('Simulation')} action={<Button icon="reset" title="Restore model parameters" onClick={onReset} />}>
      {fields.filter((field) => !field.advanced).map(renderField)}
    </Section>
    {groups.map((group) => <Section key={group} title={t(group)} initial={false}>
      {fields.filter((field) => field.advanced && field.group === group).map(renderField)}
    </Section>)}
  </div>;
}
