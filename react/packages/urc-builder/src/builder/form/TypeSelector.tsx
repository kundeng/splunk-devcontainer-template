/**
 * TypeSelector — Dropdown for choosing a component type.
 *
 * Groups supported types first, then unsupported with "(unsupported)" suffix.
 */

import React from 'react';
import Select from '@splunk/react-ui/Select';
import ControlGroup from '@splunk/react-ui/ControlGroup';

import type { ComponentFormDef } from '../../types';

export interface TypeSelectorProps {
    components: ComponentFormDef[];
    value: string;
    onChange: (type: string) => void;
}

export function TypeSelector({ components, value, onChange }: TypeSelectorProps) {
    const supported = components.filter((c) => c.supported);
    const unsupported = components.filter((c) => !c.supported);

    return (
        <ControlGroup label="Type" help="Select the component type.">
            <Select
                value={value}
                onChange={(e: any, { value: v }: any) => onChange(v as string)}
            >
                <Select.Option label="(none)" value="" />
                {supported.map((comp) => (
                    <Select.Option
                        key={comp.type}
                        label={comp.label}
                        value={comp.type}
                    />
                ))}
                {unsupported.map((comp) => (
                    <Select.Option
                        key={comp.type}
                        label={`${comp.label} (unsupported)`}
                        value={comp.type}
                    />
                ))}
            </Select>
        </ControlGroup>
    );
}

export default TypeSelector;
