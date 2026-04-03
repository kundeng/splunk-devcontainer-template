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
import Message from '@splunk/react-ui/Message';
import Key from '@splunk/react-icons/Key';

import { useBuilder } from '../../context/BuilderContext';
import { getCrossFieldWarnings } from '../../utils/validators';
import { listAccounts, listInputs, getAccountHealthMap } from '../../services/splunk-api';
import type { AccountHealth } from '../../services/splunk-api';
import { SECTIONS, FIELDS } from '../../content';
import { SectionHeader } from '../SectionHeader';
import { useFieldValidation } from '../../hooks/useFieldValidation';
import type { AccountSummary, InputSummary } from '../../types';

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

const HealthDot = styled.span<{ $status: string }>`
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    background: ${(p) =>
        p.$status === 'success'
            ? variables.accentColorPositive
            : p.$status === 'error'
              ? '#d41f1c'
              : variables.contentColorMuted};
`;

export function ConnectionTab() {
    const { state, dispatch } = useBuilder();
    const [accounts, setAccounts] = useState<AccountSummary[]>([]);
    const [loadingAccounts, setLoadingAccounts] = useState(true);
    const [existingNames, setExistingNames] = useState<string[]>([]);
    const { errors, validateField } = useFieldValidation(existingNames);
    const [accountHealth, setAccountHealth] = useState<Record<string, AccountHealth>>({});

    const isEditing = state.mode === 'edit';

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const [accts, inputs] = await Promise.all([listAccounts(), listInputs()]);
                if (!cancelled) {
                    setAccounts(accts);
                    setExistingNames(inputs.map((i: InputSummary) => i.name));
                    setAccountHealth(getAccountHealthMap());
                }
            } catch {
                // accounts/inputs may not be available yet
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
    const httpMethodWarnings = getCrossFieldWarnings(state).filter(
        (w) => w.field === 'httpMethod'
    );

    return (
        <StepContent>
            {httpMethodWarnings.map((w, i) => (
                <Message key={i} type="warning" style={{ marginBottom: 8 }}>
                    {w.message}
                </Message>
            ))}
            <SectionCard>
                <SectionHeader
                    icon={SECTIONS.streamIdentity.icon}
                    title={SECTIONS.streamIdentity.title}
                    description={SECTIONS.streamIdentity.description}
                />
                <ControlGroup
                    label={FIELDS.streamName.label}
                    help={FIELDS.streamName.help}
                    error={errors['streams[0].name']}
                >
                    <Text
                        value={state.streams[0]?.name || ''}
                        onChange={(e: any, { value }: any) => setField('streams[0].name', value)}
                        onBlur={() => validateField('streams[0].name', state.streams[0]?.name || '')}
                        placeholder={FIELDS.streamName.placeholder}
                        disabled={isEditing}
                        error={!!errors['streams[0].name']}
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
                    <div style={{ marginBottom: variables.spacingSmall, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Badge
                            label={selectedAccount.authType || 'none'}
                            icon={<Key />}
                            backgroundColor={variables.infoColor}
                            foregroundColor="#fff"
                        />
                        {accountHealth[selectedAccount.name] && (
                            <span
                                title={`Last tested: ${new Date(accountHealth[selectedAccount.name].lastTestTime).toLocaleString()}`}
                                style={{ display: 'flex', alignItems: 'center' }}
                            >
                                <HealthDot $status={accountHealth[selectedAccount.name].lastTestStatus} />
                                <span style={{ fontSize: '12px', color: variables.contentColorMuted }}>
                                    {accountHealth[selectedAccount.name].lastTestStatus === 'success'
                                        ? 'Verified'
                                        : 'Failed'}
                                </span>
                            </span>
                        )}
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
                            {accounts.map((acct) => {
                                const health = accountHealth[acct.name];
                                const suffix = health
                                    ? health.lastTestStatus === 'success'
                                        ? ' (verified)'
                                        : ' (failed)'
                                    : '';
                                return (
                                    <Select.Option
                                        key={acct.name}
                                        label={`${acct.name}${suffix}`}
                                        value={acct.name}
                                    />
                                );
                            })}
                        </Select>
                    )}
                </ControlGroup>

                <ControlGroup
                    label={FIELDS.baseUrl.label}
                    help={FIELDS.baseUrl.help}
                    error={errors['baseUrl']}
                >
                    <Text
                        value={state.baseUrl}
                        onChange={(e: any, { value }: any) => setField('baseUrl', value)}
                        onBlur={() => validateField('baseUrl', state.baseUrl)}
                        placeholder={FIELDS.baseUrl.placeholder}
                        error={!!errors['baseUrl']}
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
