/**
 * DataMappingTab — Tab 2 of the stream builder.
 *
 * Renders SchemaSection for each data-mapping aspect of the stream config:
 * Record Selector, Pagination, Incremental Sync, Transformations, Error Handling.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Button from '@splunk/react-ui/Button';
import Heading from '@splunk/react-ui/Heading';

import { useBuilder } from '../../context/BuilderContext';
import { SchemaSection } from '../form/SchemaSection';
import {
    EXTRACTORS,
    PAGINATORS,
    INCREMENTAL_CURSORS,
    TRANSFORMATIONS,
    ERROR_HANDLERS,
} from '../../schema/form-schema';

const TabContent = styled.div`
    padding: ${variables.spacingLarge} 0;
    max-width: 720px;
`;

const TransformationItem = styled.div`
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingSmall};
    margin-bottom: ${variables.spacingSmall};
    position: relative;
`;

const TransformationHeader = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${variables.spacingXSmall};
`;

const AddButton = styled.div`
    margin-top: ${variables.spacingSmall};
    margin-bottom: ${variables.spacingMedium};
`;

export function DataMappingTab() {
    const { state, dispatch } = useBuilder();
    const stream = state.streams[0];
    const validationResults = state.validationResults;

    // ── Transformations array management ──

    const transformations = stream?.transformations || [];

    const handleAddTransformation = useCallback(() => {
        const next = [...transformations, { type: '' }];
        dispatch({
            type: 'SET_FIELD',
            path: 'streams[0].transformations',
            value: next,
        });
    }, [transformations, dispatch]);

    const handleRemoveTransformation = useCallback(
        (index: number) => {
            const next = transformations.filter((_: any, i: number) => i !== index);
            dispatch({
                type: 'SET_FIELD',
                path: 'streams[0].transformations',
                value: next,
            });
        },
        [transformations, dispatch]
    );

    return (
        <TabContent>
            <SchemaSection
                title="Record Selector"
                components={EXTRACTORS}
                value={stream?.retriever?.recordSelector?.extractor || {}}
                basePath="streams[0].retriever.recordSelector.extractor"
                validationResults={validationResults}
            />

            <SchemaSection
                title="Pagination"
                components={PAGINATORS}
                value={stream?.retriever?.paginator || {}}
                basePath="streams[0].retriever.paginator"
                validationResults={validationResults}
            />

            <SchemaSection
                title="Incremental Sync"
                components={INCREMENTAL_CURSORS}
                value={stream?.incrementalSync || {}}
                basePath="streams[0].incrementalSync"
                validationResults={validationResults}
            />

            {/* Transformations — array with add/remove */}
            <Heading level={3}>Transformations</Heading>
            {transformations.map((t: Record<string, any>, idx: number) => (
                <TransformationItem key={idx}>
                    <TransformationHeader>
                        <span>Transformation {idx + 1}</span>
                        <Button
                            label="Remove"
                            appearance="destructive"
                            onClick={() => handleRemoveTransformation(idx)}
                        />
                    </TransformationHeader>
                    <SchemaSection
                        title=""
                        components={TRANSFORMATIONS}
                        value={t}
                        basePath={`streams[0].transformations[${idx}]`}
                        validationResults={validationResults}
                    />
                </TransformationItem>
            ))}
            <AddButton>
                <Button
                    label="Add Transformation"
                    appearance="secondary"
                    onClick={handleAddTransformation}
                />
            </AddButton>

            <SchemaSection
                title="Error Handling"
                components={ERROR_HANDLERS}
                value={stream?.retriever?.requester?.error_handler || {}}
                basePath="streams[0].retriever.requester.error_handler"
                validationResults={validationResults}
            />
        </TabContent>
    );
}

export default DataMappingTab;
