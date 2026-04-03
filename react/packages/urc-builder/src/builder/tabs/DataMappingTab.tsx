/**
 * DataMappingTab — Step 2: Configure how data is extracted, paginated, and transformed.
 *
 * All sections are schema-driven from form-schema.ts (auto-generated from Airbyte YAML).
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Button from '@splunk/react-ui/Button';
import Plus from '@splunk/react-icons/Plus';
import TrashCanCross from '@splunk/react-icons/TrashCanCross';

import Message from '@splunk/react-ui/Message';
import { useBuilder } from '../../context/BuilderContext';
import { SECTIONS } from '../../content';
import { getCrossFieldWarnings } from '../../utils/validators';
import { SectionHeader } from '../SectionHeader';
import { SchemaSection } from '../form/SchemaSection';
import {
    EXTRACTORS,
    PAGINATORS,
    INCREMENTAL_CURSORS,
    TRANSFORMATIONS,
    ERROR_HANDLERS,
    PARTITION_ROUTERS,
    DECODERS,
} from '../../schema/form-schema';

const TabContent = styled.div`
    max-width: 760px;
`;

const TransformationItem = styled.div`
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingSmall} ${variables.spacingMedium};
    margin-bottom: ${variables.spacingSmall};
`;

const TransformationHeader = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: ${variables.spacingXSmall};
    font-weight: 600;
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
`;

const SectionGroup = styled.div`
    margin-bottom: ${variables.spacingMedium};
`;

const AddRow = styled.div`
    margin: ${variables.spacingSmall} 0 ${variables.spacingMedium} 0;
`;

export function DataMappingTab() {
    const { state, dispatch } = useBuilder();
    const stream = state.streams[0];
    const validationResults = state.validationResults;

    const transformations = stream?.transformations || [];

    const handleAddTransformation = useCallback(() => {
        dispatch({
            type: 'SET_FIELD',
            path: 'streams[0].transformations',
            value: [...transformations, { type: '' }],
        });
    }, [transformations, dispatch]);

    const handleRemoveTransformation = useCallback(
        (index: number) => {
            dispatch({
                type: 'SET_FIELD',
                path: 'streams[0].transformations',
                value: transformations.filter((_: any, i: number) => i !== index),
            });
        },
        [transformations, dispatch]
    );

    const crossFieldWarnings = getCrossFieldWarnings(state).filter(
        (w) => w.field !== 'httpMethod' // httpMethod warnings go on ConnectionTab
    );

    return (
        <TabContent>
            {crossFieldWarnings.length > 0 && crossFieldWarnings.map((w, i) => (
                <Message key={i} type="warning" style={{ marginBottom: 8 }}>
                    {w.message}
                </Message>
            ))}
            {/* Core extraction */}
            <SchemaSection
                title="Record Selector"
                icon={SECTIONS.recordSelector.icon}
                description={SECTIONS.recordSelector.description}
                components={EXTRACTORS}
                value={stream?.retriever?.recordSelector?.extractor || {}}
                basePath="streams[0].retriever.recordSelector.extractor"
                validationResults={validationResults}
            />

            <SchemaSection
                title="Pagination"
                icon={SECTIONS.pagination.icon}
                description={SECTIONS.pagination.description}
                components={PAGINATORS}
                value={stream?.retriever?.paginator || {}}
                basePath="streams[0].retriever.paginator"
                validationResults={validationResults}
            />

            <SchemaSection
                title="Incremental Sync"
                icon={SECTIONS.incrementalSync.icon}
                description={SECTIONS.incrementalSync.description}
                components={INCREMENTAL_CURSORS}
                value={stream?.incrementalSync || {}}
                basePath="streams[0].incrementalSync"
                validationResults={validationResults}
            />

            {/* Substreams / partition routing */}
            <SchemaSection
                title="Partition Router"
                icon={SECTIONS.partitionRouter.icon}
                description={SECTIONS.partitionRouter.description}
                components={PARTITION_ROUTERS}
                value={stream?.retriever?.partition_router || {}}
                basePath="streams[0].retriever.partition_router"
                validationResults={validationResults}
            />

            {/* Response decoding */}
            <SchemaSection
                title="Decoder"
                icon={SECTIONS.decoder.icon}
                description={SECTIONS.decoder.description}
                components={DECODERS}
                value={stream?.decoder || {}}
                basePath="streams[0].decoder"
                validationResults={validationResults}
            />

            {/* Transformations — array with add/remove */}
            <SectionGroup>
                <SectionHeader
                    icon={SECTIONS.transformations.icon}
                    title={SECTIONS.transformations.title}
                    description={SECTIONS.transformations.description}
                />
                {transformations.map((t: Record<string, any>, idx: number) => (
                    <TransformationItem key={idx}>
                        <TransformationHeader>
                            <span>Transformation {idx + 1}</span>
                            <Button
                                icon={<TrashCanCross width={14} height={14} />}
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
                <AddRow>
                    <Button
                        icon={<Plus width={14} height={14} />}
                        label="Add Transformation"
                        appearance="secondary"
                        onClick={handleAddTransformation}
                    />
                </AddRow>
            </SectionGroup>

            {/* Error handling */}
            <SchemaSection
                title="Error Handling"
                icon={SECTIONS.errorHandling.icon}
                description={SECTIONS.errorHandling.description}
                components={ERROR_HANDLERS}
                value={stream?.retriever?.requester?.error_handler || {}}
                basePath="streams[0].retriever.requester.error_handler"
                validationResults={validationResults}
            />
        </TabContent>
    );
}

export default DataMappingTab;
