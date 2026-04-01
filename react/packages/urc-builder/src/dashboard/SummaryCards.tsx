/**
 * SummaryCards — displays total, active, and disabled stream counts.
 */

import React from 'react';
import styled from 'styled-components';
import ColumnLayout from '@splunk/react-ui/ColumnLayout';
import { variables } from '@splunk/themes';
import { useDashboard } from '../context/DashboardContext';

const Card = styled.div`
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingLarge} ${variables.spacingXLarge};
    text-align: center;
`;

const CardNumber = styled.div`
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.2;
    color: ${variables.contentColorDefault};
`;

const CardLabel = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-top: ${variables.spacingXSmall};
    text-transform: uppercase;
    letter-spacing: 0.05em;
`;

export default function SummaryCards() {
    const { state } = useDashboard();
    const { inputs } = state;

    const total = inputs.length;
    const active = inputs.filter((i) => !i.disabled).length;
    const disabled = inputs.filter((i) => i.disabled).length;

    return (
        <ColumnLayout>
            <ColumnLayout.Row>
                <ColumnLayout.Column span={4}>
                    <Card>
                        <CardNumber>{total}</CardNumber>
                        <CardLabel>Total Streams</CardLabel>
                    </Card>
                </ColumnLayout.Column>
                <ColumnLayout.Column span={4}>
                    <Card>
                        <CardNumber>{active}</CardNumber>
                        <CardLabel>Active</CardLabel>
                    </Card>
                </ColumnLayout.Column>
                <ColumnLayout.Column span={4}>
                    <Card>
                        <CardNumber>{disabled}</CardNumber>
                        <CardLabel>Disabled</CardLabel>
                    </Card>
                </ColumnLayout.Column>
            </ColumnLayout.Row>
        </ColumnLayout>
    );
}
