/**
 * ConnectionTab — Step 1: Stream identity + API connection settings.
 */

import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Select from '@splunk/react-ui/Select';
import Text from '@splunk/react-ui/Text';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Badge from '@splunk/react-ui/Badge';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';
import Key from '@splunk/react-icons/Key';

import { useBuilder } from '../../context/BuilderContext';
import { listAccounts } from '../../services/splunk-api';
import type { AccountSummary } from '../../types';

const StepContent = styled.div`
    max-width: 640px;
`;

const SectionCard = styled.div`
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingMedium} ${variables.spacingLarge};
    margin-bottom: ${variables.spacingMedium};
`;

const SectionHeader = styled.div`
    font-weight: 600;
    font-size: 1rem;
    color: ${variables.contentColorDefault};
    margin-bottom: ${variables.spacingMedium};
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
`;

const AccountRow = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
`;

export function ConnectionTab() {
    const { state, dispatch } = useBuilder();
    const [accounts, setAccounts] = useState<AccountSummary[]>([]);
    const [loadingAccounts, setLoadingAccounts] = useState(true);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const accts = await listAccounts();
                if (!cancelled) setAccounts(accts);
            } catch {
                // accounts may not be configured yet
            } finally {
                if (!cancelled) setLoadingAccounts(false);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const setField = useCallback(
        (path: string, value: any) => dispatch({ type: 'SET_FIELD', path, value }),
        [dispatch]
    );

    const selectedAccount = accounts.find((a) => a.name === state.account);

    return (
        <StepContent>
            <SectionCard>
                <SectionHeader>Stream Identity</SectionHeader>
                <ControlGroup label="Stream Name" help="A unique identifier for this data stream (letters, numbers, underscores).">
                    <Text
                        value={state.streams[0]?.name || ''}
                        onChange={(e: any, { value }: any) => {
                            setField('streams[0].name', value);
                        }}
                        placeholder="e.g. github_issues, jira_tickets"
                    />
                </ControlGroup>
            </SectionCard>

            <SectionCard>
                <SectionHeader>
                    API Connection
                    {selectedAccount && (
                        <Badge
                            label={selectedAccount.authType || 'none'}
                            icon={<Key />}
                            backgroundColor={variables.infoColor}
                            foregroundColor="#fff"
                        />
                    )}
                </SectionHeader>

                <ControlGroup label="Account" help="Select the authentication account for this API. Create accounts in Configuration.">
                    {loadingAccounts ? (
                        <WaitSpinner />
                    ) : (
                        <Select
                            value={state.account}
                            onChange={(e: any, { value }: any) => setField('account', value)}
                        >
                            <Select.Option label="(none — no authentication)" value="" />
                            {accounts.map((acct) => (
                                <Select.Option
                                    key={acct.name}
                                    label={acct.name}
                                    value={acct.name}
                                />
                            ))}
                        </Select>
                    )}
                </ControlGroup>

                <ControlGroup label="Base URL" help="The root URL of the API endpoint to collect from.">
                    <Text
                        value={state.baseUrl}
                        onChange={(e: any, { value }: any) => setField('baseUrl', value)}
                        placeholder="https://api.example.com/v1/resources"
                    />
                </ControlGroup>

                <ControlGroup label="HTTP Method" help="HTTP method for API requests. Use POST only for APIs that require it.">
                    <Select
                        value={state.httpMethod}
                        onChange={(e: any, { value }: any) => setField('httpMethod', value)}
                    >
                        <Select.Option label="GET" value="GET" />
                        <Select.Option label="POST" value="POST" />
                    </Select>
                </ControlGroup>
            </SectionCard>
        </StepContent>
    );
}

export default ConnectionTab;
