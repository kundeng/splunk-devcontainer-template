/**
 * SchemaSection — Collapsible section that renders a schema-driven form.
 *
 * Shows a TypeSelector to pick the component type, then renders
 * SchemaField for each field of the selected type definition.
 */

import React, { useCallback, useMemo } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import CollapsiblePanel from '@splunk/react-ui/CollapsiblePanel';
import Chip from '@splunk/react-ui/Chip';
import Tooltip from '@splunk/react-ui/Tooltip';
import QuestionCircle from '@splunk/react-icons/QuestionCircle';

import { useBuilder } from '../../context/BuilderContext';
import type { ComponentFormDef, ValidationResult } from '../../types';
import { TypeSelector } from './TypeSelector';
import { SchemaField } from './SchemaField';
import { ValidationBanner } from './ValidationBanner';

const PanelTitle = styled.span`
    display: inline-flex;
    align-items: center;
    gap: 8px;
`;

const TitleIcon = styled.span`
    display: inline-flex;
    align-items: center;
    color: ${variables.infoColor};
`;

const HelpTrigger = styled.span`
    display: inline-flex;
    align-items: center;
    color: ${variables.contentColorMuted};
    cursor: help;
    margin-left: 2px;
`;

export interface SchemaSectionProps {
    /** Display title for the collapsible section */
    title: string;
    /** Icon element for the section header */
    icon?: React.ReactNode;
    /** Help text shown in tooltip */
    description?: string;
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
    icon,
    description,
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

    const panelTitle = (
        <PanelTitle>
            {icon && <TitleIcon>{icon}</TitleIcon>}
            {title}
            {currentType && (
                <Chip appearance="info">{currentType}</Chip>
            )}
            {description && (
                <Tooltip content={description}>
                    <HelpTrigger><QuestionCircle width={14} height={14} /></HelpTrigger>
                </Tooltip>
            )}
        </PanelTitle>
    );

    return (
        <CollapsiblePanel title={title ? panelTitle : undefined}>
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
