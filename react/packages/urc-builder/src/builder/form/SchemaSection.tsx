/**
 * SchemaSection — Collapsible section that renders a schema-driven form.
 *
 * Shows a TypeSelector to pick the component type, then renders
 * SchemaField for each field of the selected type definition.
 */

import React, { useCallback, useMemo } from 'react';
import CollapsiblePanel from '@splunk/react-ui/CollapsiblePanel';

import { useBuilder } from '../../context/BuilderContext';
import type { ComponentFormDef, ValidationResult } from '../../types';
import { TypeSelector } from './TypeSelector';
import { SchemaField } from './SchemaField';
import { ValidationBanner } from './ValidationBanner';

export interface SchemaSectionProps {
    /** Display title for the collapsible section */
    title: string;
    /** Array of ComponentFormDef from form-schema.ts */
    components: ComponentFormDef[];
    /** Current config object for this section */
    value: Record<string, any>;
    /** Dot-path prefix for dispatching SET_FIELD (e.g. "streams[0].retriever.paginator") */
    basePath: string;
    /** Validation results to filter and show */
    validationResults?: ValidationResult[];
}

export function SchemaSection({
    title,
    components,
    value,
    basePath,
    validationResults = [],
}: SchemaSectionProps) {
    const { dispatch } = useBuilder();

    const currentType = value?.type || '';

    const selectedDef = useMemo(
        () => components.find((c) => c.type === currentType),
        [components, currentType]
    );

    const handleTypeChange = useCallback(
        (newType: string) => {
            // Find the new definition and build default values
            const def = components.find((c) => c.type === newType);
            const defaults: Record<string, any> = { type: newType };
            if (def) {
                for (const field of def.fields) {
                    if (field.defaultValue !== undefined) {
                        defaults[field.key] = field.defaultValue;
                    }
                }
            }
            dispatch({ type: 'SET_FIELD', path: basePath, value: defaults });
        },
        [components, basePath, dispatch]
    );

    const handleFieldChange = useCallback(
        (fieldPath: string, fieldValue: any) => {
            dispatch({ type: 'SET_FIELD', path: `${basePath}.${fieldPath}`, value: fieldValue });
        },
        [basePath, dispatch]
    );

    return (
        <CollapsiblePanel title={title}>
            <ValidationBanner results={validationResults} pathPrefix={basePath} />

            <TypeSelector
                components={components}
                value={currentType}
                onChange={handleTypeChange}
            />

            {selectedDef &&
                selectedDef.fields.map((field) => (
                    <SchemaField
                        key={field.key}
                        field={field}
                        value={value?.[field.key]}
                        onChange={(val) => handleFieldChange(field.key, val)}
                    />
                ))}
        </CollapsiblePanel>
    );
}

export default SchemaSection;
