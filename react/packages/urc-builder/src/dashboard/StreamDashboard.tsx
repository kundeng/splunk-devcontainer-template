/**
 * StreamDashboard — top-level container for the Stream Dashboard.
 *
 * Wraps child components in DashboardProvider, fetches inputs on mount,
 * and orchestrates layout of summary cards, filters, bulk actions, and table.
 */

import React, { useEffect, useMemo, useState, useCallback } from 'react';
import styled from 'styled-components';
import Button from '@splunk/react-ui/Button';
import Heading from '@splunk/react-ui/Heading';
import Paragraph from '@splunk/react-ui/Paragraph';
import Message from '@splunk/react-ui/Message';
import Divider from '@splunk/react-ui/Divider';
import { variables } from '@splunk/themes';
import Plus from '@splunk/react-icons/Plus';
import { DASHBOARD } from '../content';
import { DashboardProvider, useDashboard } from '../context/DashboardContext';
import { listInputs, getStreamHealth } from '../services/splunk-api';
import { useGroupBy } from '../hooks/useGroupBy';
import SummaryCards from './SummaryCards';
import TagFilter from './TagFilter';
import BulkActions from './BulkActions';
import StreamTable from './StreamTable';
import ImportModal from './ImportModal';

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

const GroupHeader = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
    padding: ${variables.spacingSmall} 0;
    margin-top: ${variables.spacingMedium};
    border-bottom: 1px solid ${variables.borderColor};
`;

const GroupLabel = styled.span`
    font-weight: 600;
    font-size: ${variables.fontSizeSmall};
`;

const GroupCount = styled.span`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
`;

function DashboardInner() {
    const { setInputs, setError, state, filteredInputs, groupBy, setHealthMap } = useDashboard();
    const { groups } = useGroupBy(filteredInputs, groupBy);
    const [importOpen, setImportOpen] = useState(false);

    const handleImportComplete = useCallback(async () => {
        try {
            const inputs = await listInputs();
            setInputs(inputs);
        } catch { /* ignore */ }
        setImportOpen(false);
    }, [setInputs]);

    useEffect(() => {
        let cancelled = false;

        async function fetchData() {
            try {
                const [inputs, healthList] = await Promise.all([
                    listInputs(),
                    getStreamHealth(),
                ]);
                if (!cancelled) {
                    setInputs(inputs);
                    const map: Record<string, any> = {};
                    for (const h of healthList) {
                        map[h.name] = h;
                    }
                    setHealthMap(map);
                }
            } catch (err: any) {
                if (!cancelled) {
                    setError(err.message || 'Failed to load inputs');
                }
            }
        }

        fetchData();
        return () => { cancelled = true; };
    }, [setInputs, setError, setHealthMap]);

    return (
        <PageWrapper>
            <PageHeader>
                <HeaderLeft>
                    <Heading level={1}>{DASHBOARD.title}</Heading>
                    <Subtitle>{DASHBOARD.subtitle}</Subtitle>
                </HeaderLeft>
                <div style={{ display: 'flex', gap: 8 }}>
                    <Button
                        appearance="secondary"
                        label="Import"
                        onClick={() => setImportOpen(true)}
                    />
                    <Button
                        appearance="primary"
                        icon={<Plus />}
                        label={DASHBOARD.newStreamCta}
                        onClick={() => { window.location.href = 'builder'; }}
                    />
                </div>
            </PageHeader>

            <ImportModal
                open={importOpen}
                onClose={() => setImportOpen(false)}
                existingInputs={state.inputs}
                onComplete={handleImportComplete}
            />

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

            {groups ? (
                groups.map((group) => (
                    <Section key={group.label}>
                        <GroupHeader>
                            <GroupLabel>{group.label}</GroupLabel>
                            <GroupCount>({group.items.length})</GroupCount>
                        </GroupHeader>
                        <StreamTable items={group.items} />
                    </Section>
                ))
            ) : (
                <Section>
                    <StreamTable />
                </Section>
            )}
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
