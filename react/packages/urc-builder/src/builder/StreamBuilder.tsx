/**
 * StreamBuilder — Top-level builder container.
 *
 * Wraps the entire builder form in BuilderProvider, handles loading
 * existing inputs, tab navigation, and save/cancel actions.
 */

import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import TabLayout from '@splunk/react-ui/TabLayout';
import Button from '@splunk/react-ui/Button';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';
import Message from '@splunk/react-ui/Message';

import { BuilderProvider, useBuilder } from '../context/BuilderContext';
import { getInput, createInput, updateInput } from '../services/splunk-api';
import { formStateToManifest } from '../services/manifest-serializer';
import type { InputPayload } from '../types';
import { ConnectionTab } from './tabs/ConnectionTab';
import { DataMappingTab } from './tabs/DataMappingTab';

// ── Styled wrappers ──

const Container = styled.div`
    padding: ${variables.spacingLarge};
`;

const Footer = styled.div`
    position: sticky;
    bottom: 0;
    display: flex;
    gap: ${variables.spacingSmall};
    justify-content: flex-end;
    padding: ${variables.spacingMedium} ${variables.spacingLarge};
    background: ${variables.backgroundColorSection};
    border-top: 1px solid ${variables.borderColor};
    z-index: 10;
`;

const CenteredSpinner = styled.div`
    display: flex;
    justify-content: center;
    padding: ${variables.spacingXXLarge};
`;

// ── Inner component (needs BuilderProvider above it) ──

function StreamBuilderInner({ inputName }: { inputName?: string }) {
    const { state, dispatch } = useBuilder();
    const [activeTab, setActiveTab] = useState('connection');
    const [loading, setLoading] = useState(!!inputName);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Load existing input on mount
    useEffect(() => {
        if (!inputName) return;
        let cancelled = false;

        (async () => {
            try {
                const input = await getInput(inputName);
                if (cancelled) return;
                dispatch({
                    type: 'LOAD_INPUT',
                    input: { ...input, parsedManifest: {} },
                });
            } catch (err: any) {
                if (!cancelled) setError(err.message || 'Failed to load input');
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();

        return () => { cancelled = true; };
    }, [inputName, dispatch]);

    const handleSave = useCallback(async () => {
        setSaving(true);
        setError(null);

        try {
            const manifest = formStateToManifest(state);
            const payload: InputPayload = {
                name: state.inputName || state.streams[0]?.name || 'default',
                account: state.account,
                base_url: state.baseUrl,
                manifest,
                interval: String(state.interval),
                index: state.index,
                sourcetype: state.sourcetype,
                tags: state.tags,
                debug: state.debug ? '1' : '0',
                disabled: state.enabled ? '0' : '1',
            };

            if (state.mode === 'edit' && state.inputName) {
                await updateInput(state.inputName, payload);
            } else {
                await createInput(payload);
            }

            window.location.href = 'streams';
        } catch (err: any) {
            setError(err.message || 'Save failed');
        } finally {
            setSaving(false);
        }
    }, [state]);

    const handleCancel = useCallback(() => {
        window.location.href = 'streams';
    }, []);

    if (loading) {
        return (
            <CenteredSpinner>
                <WaitSpinner size="large" />
            </CenteredSpinner>
        );
    }

    return (
        <Container>
            {error && (
                <Message type="error" onRequestRemove={() => setError(null)}>
                    {error}
                </Message>
            )}

            <TabLayout
                activePanelId={activeTab}
                onChange={(e: any, { activePanelId }: any) => setActiveTab(activePanelId)}
            >
                <TabLayout.Panel label="Connection" panelId="connection">
                    <ConnectionTab />
                </TabLayout.Panel>
                <TabLayout.Panel label="Data Mapping" panelId="data-mapping">
                    <DataMappingTab />
                </TabLayout.Panel>
                <TabLayout.Panel label="Splunk Output" panelId="splunk-output">
                    <div>Splunk Output tab — coming soon</div>
                </TabLayout.Panel>
                <TabLayout.Panel label="Schedule & Tags" panelId="schedule-tags">
                    <div>Schedule & Tags tab — coming soon</div>
                </TabLayout.Panel>
                <TabLayout.Panel label="Test & Preview" panelId="test-preview">
                    <div>Test & Preview tab — coming soon</div>
                </TabLayout.Panel>
            </TabLayout>

            <Footer>
                <Button label="Cancel" onClick={handleCancel} />
                <Button
                    label={saving ? 'Saving...' : 'Save'}
                    appearance="primary"
                    onClick={handleSave}
                    disabled={saving}
                />
            </Footer>
        </Container>
    );
}

// ── Public component ──

export interface StreamBuilderProps {
    inputName?: string;
}

export function StreamBuilder({ inputName }: StreamBuilderProps) {
    return (
        <BuilderProvider>
            <StreamBuilderInner inputName={inputName} />
        </BuilderProvider>
    );
}

export default StreamBuilder;
