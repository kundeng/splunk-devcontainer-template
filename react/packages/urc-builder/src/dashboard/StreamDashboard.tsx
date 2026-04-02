/**
 * StreamDashboard — top-level container for the Stream Dashboard.
 *
 * Wraps child components in DashboardProvider, fetches inputs on mount,
 * and orchestrates layout of summary cards, filters, bulk actions, and table.
 */

import React, { useEffect } from 'react';
import styled from 'styled-components';
import Button from '@splunk/react-ui/Button';
import Heading from '@splunk/react-ui/Heading';
import Paragraph from '@splunk/react-ui/Paragraph';
import Message from '@splunk/react-ui/Message';
import Divider from '@splunk/react-ui/Divider';
import { variables } from '@splunk/themes';
import Plus from '@splunk/react-icons/Plus';
import { DashboardProvider, useDashboard } from '../context/DashboardContext';
import { listInputs } from '../services/splunk-api';
import SummaryCards from './SummaryCards';
import TagFilter from './TagFilter';
import BulkActions from './BulkActions';
import StreamTable from './StreamTable';

const PageWrapper = styled.div`
    max-width: 1200px;
    margin: 0 auto;
    padding: ${variables.spacingLarge};
`;

const PageHeader = styled.div`
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: ${variables.spacingMedium};
`;

const HeaderLeft = styled.div``;

const Subtitle = styled.p`
    color: ${variables.contentColorMuted};
    font-size: ${variables.fontSizeSmall};
    margin: 4px 0 0 0;
`;

const Section = styled.div`
    margin-bottom: ${variables.spacingMedium};
`;

function DashboardInner() {
    const { setInputs, setError, state } = useDashboard();

    useEffect(() => {
        let cancelled = false;

        async function fetchInputs() {
            try {
                const inputs = await listInputs();
                if (!cancelled) {
                    setInputs(inputs);
                }
            } catch (err: any) {
                if (!cancelled) {
                    setError(err.message || 'Failed to load inputs');
                }
            }
        }

        fetchInputs();
        return () => { cancelled = true; };
    }, [setInputs, setError]);

    return (
        <PageWrapper>
            <PageHeader>
                <HeaderLeft>
                    <Heading level={1}>REST API Streams</Heading>
                    <Subtitle>Configure and monitor data collection from REST APIs</Subtitle>
                </HeaderLeft>
                <Button
                    appearance="primary"
                    icon={<Plus />}
                    label="New Stream"
                    onClick={() => { window.location.href = 'builder'; }}
                />
            </PageHeader>

            <Divider />

            {state.error && (
                <Section>
                    <Message type="error">{state.error}</Message>
                </Section>
            )}

            <Section>
                <SummaryCards />
            </Section>

            <TagFilter />

            <BulkActions />

            <Section>
                <StreamTable />
            </Section>
        </PageWrapper>
    );
}

export default function StreamDashboard() {
    return (
        <DashboardProvider>
            <DashboardInner />
        </DashboardProvider>
    );
}
