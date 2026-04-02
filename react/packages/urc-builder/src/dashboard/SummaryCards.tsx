/**
 * SummaryCards — Visual summary of stream counts with colored accents and icons.
 */

import React from 'react';
import styled from 'styled-components';
import ColumnLayout from '@splunk/react-ui/ColumnLayout';
import { variables } from '@splunk/themes';
import { DASHBOARD } from '../content';
import { useDashboard } from '../context/DashboardContext';

interface AccentProps {
    $color: string;
}

const CardAccent = styled.div<AccentProps>`
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-left: 4px solid ${(p) => p.$color};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingMedium} ${variables.spacingLarge};
    display: flex;
    align-items: center;
    gap: ${variables.spacingMedium};
`;

const IconWrap = styled.div<AccentProps>`
    color: ${(p) => p.$color};
    display: flex;
    align-items: center;
    opacity: 0.7;
    & svg {
        width: 28px;
        height: 28px;
    }
`;

const CardContent = styled.div``;

const CardNumber = styled.div`
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
    color: ${variables.contentColorDefault};
`;

const CardLabel = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
`;

export default function SummaryCards() {
    const { state } = useDashboard();
    const total = state.inputs.length;
    const active = state.inputs.filter((i) => !i.disabled).length;
    const disabled = state.inputs.filter((i) => i.disabled).length;

    const cards = [
        { ...DASHBOARD.cards.total, value: total, color: variables.infoColor },
        { ...DASHBOARD.cards.active, value: active, color: variables.accentColorPositive },
        { ...DASHBOARD.cards.disabled, value: disabled, color: variables.contentColorMuted },
    ];

    return (
        <ColumnLayout gutter={12}>
            <ColumnLayout.Row>
                {cards.map((card) => (
                    <ColumnLayout.Column key={card.label} span={4}>
                        <CardAccent $color={card.color}>
                            <IconWrap $color={card.color}>{card.icon}</IconWrap>
                            <CardContent>
                                <CardNumber>{card.value}</CardNumber>
                                <CardLabel>{card.label}</CardLabel>
                            </CardContent>
                        </CardAccent>
                    </ColumnLayout.Column>
                ))}
            </ColumnLayout.Row>
        </ColumnLayout>
    );
}
