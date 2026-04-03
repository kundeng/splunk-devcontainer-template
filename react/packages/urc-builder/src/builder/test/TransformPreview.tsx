import React from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Heading from '@splunk/react-ui/Heading';
import Code from '@splunk/react-ui/Code';
import Message from '@splunk/react-ui/Message';

const PreviewGrid = styled.div`
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: ${variables.spacingMedium};
    margin-top: ${variables.spacingMedium};
`;

const PreviewPanel = styled.div`
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingSmall};
`;

const PanelLabel = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    font-weight: 600;
    margin-bottom: ${variables.spacingXSmall};
    text-transform: uppercase;
`;

interface TransformPreviewProps {
    rawRecords: { stream: string; data: Record<string, any> }[];
    transformedRecords: { stream: string; data: Record<string, any> }[];
    hasTransformations: boolean;
}

export function TransformPreview({ rawRecords, transformedRecords, hasTransformations }: TransformPreviewProps) {
    if (!hasTransformations || rawRecords.length === 0) {
        return null;
    }

    const rawSample = rawRecords[0]?.data || {};
    const transformedSample = transformedRecords[0]?.data || rawSample;

    return (
        <div>
            <Heading level={4}>Transformation Preview</Heading>
            <PreviewGrid>
                <PreviewPanel>
                    <PanelLabel>Raw</PanelLabel>
                    <Code language="json" value={JSON.stringify(rawSample, null, 2)} />
                </PreviewPanel>
                <PreviewPanel>
                    <PanelLabel>After Transformations</PanelLabel>
                    <Code language="json" value={JSON.stringify(transformedSample, null, 2)} />
                </PreviewPanel>
            </PreviewGrid>
            <Message type="info" style={{ marginTop: 8 }}>
                Transformation preview shows raw API response vs. test result. Differences indicate transformations are active.
            </Message>
        </div>
    );
}
