/**
 * FieldDetector — lists detected fields from sample records with types.
 */

import React from 'react';
import Table from '@splunk/react-ui/Table';
import Heading from '@splunk/react-ui/Heading';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import type { DetectedField } from '../../types';

const TypeBadge = styled.span<{ fieldType: string }>`
    padding: 2px 6px;
    border-radius: 3px;
    font-size: ${variables.fontSizeSmall};
    font-family: monospace;
    background: ${(p) =>
        p.fieldType === 'datetime'
            ? variables.accentColorL10
            : p.fieldType === 'number'
              ? variables.infoColor
              : 'transparent'};
`;

interface FieldDetectorProps {
    fields: DetectedField[];
}

export const FieldDetector: React.FC<FieldDetectorProps> = ({ fields }) => {
    if (fields.length === 0) return null;

    return (
        <div>
            <Heading level={4}>Detected Fields ({fields.length})</Heading>
            <Table stripeRows>
                <Table.Head>
                    <Table.HeadCell>Field</Table.HeadCell>
                    <Table.HeadCell>Type</Table.HeadCell>
                </Table.Head>
                <Table.Body>
                    {fields.map((f) => (
                        <Table.Row key={f.name}>
                            <Table.Cell style={{ fontFamily: 'monospace' }}>{f.name}</Table.Cell>
                            <Table.Cell>
                                <TypeBadge fieldType={f.type}>{f.type}</TypeBadge>
                            </Table.Cell>
                        </Table.Row>
                    ))}
                </Table.Body>
            </Table>
        </div>
    );
};
