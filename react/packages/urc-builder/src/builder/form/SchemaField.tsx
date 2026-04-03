/**
 * SchemaField — Renders a single form field based on FormFieldDef.type.
 *
 * Dispatches by type: text, number, boolean, enum, template, object, array.
 * Each field is wrapped in a ControlGroup with label and helpText.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Text from '@splunk/react-ui/Text';
import Select from '@splunk/react-ui/Select';
import Switch from '@splunk/react-ui/Switch';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Button from '@splunk/react-ui/Button';

import type { FormFieldDef } from '../../types';
import { TemplatePreview } from './TemplatePreview';

const NestedFieldset = styled.fieldset`
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingSmall} ${variables.spacingMedium};
    margin: ${variables.spacingXSmall} 0;
`;

const NestedLegend = styled.legend`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    padding: 0 ${variables.spacingXSmall};
`;

const ArrayRow = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingXSmall};
    margin-bottom: ${variables.spacingXSmall};
`;

const ArrayControls = styled.div`
    margin-top: ${variables.spacingXSmall};
`;

export interface SchemaFieldProps {
    field: FormFieldDef;
    value: any;
    onChange: (value: any) => void;
}

export function SchemaField({ field, value, onChange }: SchemaFieldProps) {
    const label = field.required ? `${field.label} *` : field.label;

    switch (field.type) {
        case 'text':
            return (
                <ControlGroup label={label} help={field.helpText}>
                    <Text
                        value={value ?? field.defaultValue ?? ''}
                        onChange={(e: any, { value: v }: any) => onChange(v)}
                        placeholder={field.examples?.[0] || ''}
                    />
                </ControlGroup>
            );

        case 'number':
            return (
                <ControlGroup label={label} help={field.helpText}>
                    <Text
                        type="number"
                        value={value ?? field.defaultValue ?? ''}
                        onChange={(e: any, { value: v }: any) => onChange(Number(v))}
                    />
                </ControlGroup>
            );

        case 'boolean':
            return (
                <ControlGroup label={label} help={field.helpText}>
                    <Switch
                        selected={value ?? field.defaultValue ?? false}
                        onClick={() => onChange(!value)}
                        appearance="toggle"
                    />
                </ControlGroup>
            );

        case 'enum':
            return (
                <ControlGroup label={label} help={field.helpText}>
                    <Select
                        value={value ?? field.defaultValue ?? ''}
                        onChange={(e: any, { value: v }: any) => onChange(v)}
                    >
                        <Select.Option label="(select)" value="" />
                        {(field.enumValues || []).map((opt) => (
                            <Select.Option
                                key={opt.value}
                                label={opt.label}
                                value={opt.value}
                            />
                        ))}
                    </Select>
                </ControlGroup>
            );

        case 'template':
            return (
                <ControlGroup
                    label={`${label} {{ }}`}
                    help={field.helpText}
                >
                    <Text
                        value={value ?? field.defaultValue ?? ''}
                        onChange={(e: any, { value: v }: any) => onChange(v)}
                        placeholder={field.examples?.[0] || ''}
                    />
                    <TemplatePreview
                        template={value ?? ''}
                        context={field.interpolationContext?.reduce((acc: Record<string, string>, v: string) => ({ ...acc, [v]: `<${v}>` }), {}) || {}}
                    />
                </ControlGroup>
            );

        case 'object':
            return <ObjectField field={field} value={value} onChange={onChange} label={label} />;

        case 'array':
            return <ArrayField field={field} value={value} onChange={onChange} label={label} />;

        default:
            return null;
    }
}

// ── Object field with nested fields ──

function ObjectField({
    field,
    value,
    onChange,
    label,
}: SchemaFieldProps & { label: string }) {
    const obj = value && typeof value === 'object' ? value : {};

    const handleNestedChange = useCallback(
        (key: string, val: any) => {
            onChange({ ...obj, [key]: val });
        },
        [obj, onChange]
    );

    if (!field.nestedFields || field.nestedFields.length === 0) {
        // Fallback: render as JSON text
        return (
            <ControlGroup label={label} help={field.helpText}>
                <Text
                    value={typeof value === 'object' ? JSON.stringify(value, null, 2) : value ?? ''}
                    onChange={(e: any, { value: v }: any) => {
                        try { onChange(JSON.parse(v)); } catch { onChange(v); }
                    }}
                />
            </ControlGroup>
        );
    }

    return (
        <NestedFieldset>
            <NestedLegend>{label}</NestedLegend>
            {field.nestedFields.map((nested) => (
                <SchemaField
                    key={nested.key}
                    field={nested}
                    value={obj[nested.key]}
                    onChange={(val) => handleNestedChange(nested.key, val)}
                />
            ))}
        </NestedFieldset>
    );
}

// ── Array field with add/remove ──

function ArrayField({
    field,
    value,
    onChange,
    label,
}: SchemaFieldProps & { label: string }) {
    const items: any[] = Array.isArray(value) ? value : [];

    const handleItemChange = useCallback(
        (index: number, val: any) => {
            const next = [...items];
            next[index] = val;
            onChange(next);
        },
        [items, onChange]
    );

    const handleAdd = useCallback(() => {
        onChange([...items, '']);
    }, [items, onChange]);

    const handleRemove = useCallback(
        (index: number) => {
            onChange(items.filter((_, i) => i !== index));
        },
        [items, onChange]
    );

    return (
        <ControlGroup label={label} help={field.helpText}>
            {items.map((item, idx) => (
                <ArrayRow key={idx}>
                    <Text
                        value={typeof item === 'string' ? item : JSON.stringify(item)}
                        onChange={(e: any, { value: v }: any) => handleItemChange(idx, v)}
                        style={{ flex: 1 }}
                    />
                    <Button
                        label="Remove"
                        appearance="destructive"
                        onClick={() => handleRemove(idx)}
                    />
                </ArrayRow>
            ))}
            <ArrayControls>
                <Button label="Add" appearance="secondary" onClick={handleAdd} />
            </ArrayControls>
        </ControlGroup>
    );
}

export default SchemaField;
