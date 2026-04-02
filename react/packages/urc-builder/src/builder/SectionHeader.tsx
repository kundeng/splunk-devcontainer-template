/**
 * SectionHeader — Section header with aligned icon, title, and help tooltip.
 */

import React from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Tooltip from '@splunk/react-ui/Tooltip';
import QuestionCircle from '@splunk/react-icons/QuestionCircle';

const Header = styled.div`
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: ${variables.spacingMedium};
`;

const IconWrap = styled.span`
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: ${variables.infoColor};
    flex-shrink: 0;
`;

const Title = styled.span`
    font-weight: 600;
    font-size: 1rem;
    color: ${variables.contentColorDefault};
`;

const HelpTrigger = styled.span`
    display: inline-flex;
    align-items: center;
    color: ${variables.contentColorMuted};
    cursor: help;
    margin-left: 4px;
`;

interface SectionHeaderProps {
    icon: React.ReactNode;
    title: string;
    description?: string;
}

export function SectionHeader({ icon, title, description }: SectionHeaderProps) {
    return (
        <Header>
            <IconWrap>{icon}</IconWrap>
            <Title>{title}</Title>
            {description && (
                <Tooltip content={description}>
                    <HelpTrigger>
                        <QuestionCircle width={14} height={14} />
                    </HelpTrigger>
                </Tooltip>
            )}
        </Header>
    );
}

export default SectionHeader;
