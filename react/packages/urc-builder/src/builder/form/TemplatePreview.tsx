import React, { useMemo } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';

const PreviewLine = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-top: 4px;
    padding: 4px 8px;
    background: ${variables.backgroundColorPage};
    border-radius: ${variables.borderRadius};
    font-family: monospace;
`;

const UnresolvedVar = styled.span`
    color: #d41f1c;
    font-weight: 600;
`;

interface TemplatePreviewProps {
    template: string;
    context?: Record<string, string>;
}

export function TemplatePreview({ template, context = {} }: TemplatePreviewProps) {
    const resolved = useMemo(() => {
        if (!template || !template.includes('{{')) return null;

        return template.replace(/\{\{\s*(\w+(?:\.\w+)*)\s*\}\}/g, (match, varName) => {
            if (context[varName] !== undefined) {
                return context[varName];
            }
            return `__UNRESOLVED__${varName}__`;
        });
    }, [template, context]);

    if (!resolved) return null;

    // Split into parts for rendering with styled unresolved vars
    const parts = resolved.split(/__UNRESOLVED__(\w+(?:\.\w+)*)__/);

    return (
        <PreviewLine>
            Preview:{' '}
            {parts.map((part, i) =>
                i % 2 === 0 ? (
                    <span key={i}>{part}</span>
                ) : (
                    <UnresolvedVar key={i}>{`{{ ${part} }}`}</UnresolvedVar>
                )
            )}
        </PreviewLine>
    );
}
