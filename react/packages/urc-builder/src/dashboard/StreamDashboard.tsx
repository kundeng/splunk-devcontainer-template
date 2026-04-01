/**
 * StreamDashboard — top-level container for the Stream Dashboard (Screen 1).
 *
 * Wraps child components in DashboardProvider, fetches inputs on mount,
 * and orchestrates layout of summary cards, filters, bulk actions, and table.
 */

import React, { useEffect } from 'react';
import styled from 'styled-components';
import Heading from '@splunk/react-ui/Heading';
import Message from '@splunk/react-ui/Message';
import { variables } from '@splunk/themes';
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

const Section = styled.div`
    margin-bottom: ${variables.spacingMedium};
`;

/**
 * Inner component that consumes DashboardContext.
 * Separated so that useEffect can call context setters.
 */
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

        return () => {
            cancelled = true;
        };
    }, [setInputs, setError]);

    return (
        <PageWrapper>
            <Section>
                <Heading level={1}>REST API Streams</Heading>
            </Section>

            {state.error && (
                <Section>
                    <Message type="error">{state.error}</Message>
                </Section>
            )}

            <Section>
                <SummaryCards />
            </Section>

            <Section>
                <TagFilter />
            </Section>

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
