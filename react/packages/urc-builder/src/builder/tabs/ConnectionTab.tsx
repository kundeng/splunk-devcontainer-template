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
import { SECTIONS, FIELDS } from '../../content';
import { SectionHeader } from '../SectionHeader';
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

const AuthBadgeWrap = styled.div`
    margin-left: auto;
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
                <SectionHeader
                    icon={SECTIONS.streamIdentity.icon}
                    title={SECTIONS.streamIdentity.title}
                    description={SECTIONS.streamIdentity.description}
                />
                <ControlGroup label={FIELDS.streamName.label} help={FIELDS.streamName.help}>
                    <Text
                        value={state.streams[0]?.name || ''}
                        onChange={(e: any, { value }: any) => setField('streams[0].name', value)}
                        placeholder={FIELDS.streamName.placeholder}
                    />
                </ControlGroup>
            </SectionCard>

            <SectionCard>
                <SectionHeader
                    icon={SECTIONS.apiConnection.icon}
                    title={SECTIONS.apiConnection.title}
                    description={SECTIONS.apiConnection.description}
                />

                {selectedAccount && (
                    <div style={{ marginBottom: variables.spacingSmall }}>
                        <Badge
                            label={selectedAccount.authType || 'none'}
                            icon={<Key />}
                            backgroundColor={variables.infoColor}
                            foregroundColor="#fff"
                        />
                    </div>
                )}

                <ControlGroup label={FIELDS.account.label} help={FIELDS.account.help}>
                    {loadingAccounts ? (
                        <WaitSpinner />
                    ) : (
                        <Select
                            value={state.account}
                            onChange={(e: any, { value }: any) => setField('account', value)}
                        >
                            <Select.Option label={FIELDS.account.placeholder!} value="" />
                            {accounts.map((acct) => (
                                <Select.Option key={acct.name} label={acct.name} value={acct.name} />
                            ))}
                        </Select>
                    )}
                </ControlGroup>

                <ControlGroup label={FIELDS.baseUrl.label} help={FIELDS.baseUrl.help}>
                    <Text
                        value={state.baseUrl}
                        onChange={(e: any, { value }: any) => setField('baseUrl', value)}
                        placeholder={FIELDS.baseUrl.placeholder}
                    />
                </ControlGroup>

                <ControlGroup label={FIELDS.httpMethod.label} help={FIELDS.httpMethod.help}>
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
