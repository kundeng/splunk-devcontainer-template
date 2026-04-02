/**
 * SectionHeader — Reusable section header with icon, title, and description.
 */

import React from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Tooltip from '@splunk/react-ui/Tooltip';
import QuestionCircle from '@splunk/react-icons/QuestionCircle';

const Header = styled.div`
    display: flex;
    align-items: flex-start;
    gap: ${variables.spacingSmall};
    margin-bottom: ${variables.spacingMedium};
`;

const IconWrap = styled.div`
    color: ${variables.infoColor};
    font-size: 20px;
    margin-top: 2px;
    flex-shrink: 0;
`;

const Content = styled.div`
    flex: 1;
`;

const TitleRow = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingXSmall};
`;

const Title = styled.div`
    font-weight: 600;
    font-size: 1rem;
    color: ${variables.contentColorDefault};
`;

const HelpIcon = styled.span`
    color: ${variables.contentColorMuted};
    font-size: 14px;
    cursor: help;
    display: inline-flex;
    align-items: center;
`;

const Description = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-top: 2px;
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
            <Content>
                <TitleRow>
                    <Title>{title}</Title>
                    {description && (
                        <Tooltip content={description}>
                            <HelpIcon><QuestionCircle /></HelpIcon>
                        </Tooltip>
                    )}
                </TitleRow>
                {description && <Description>{description}</Description>}
            </Content>
        </Header>
    );
}

export default SectionHeader;
