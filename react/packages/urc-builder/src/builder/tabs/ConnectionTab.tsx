/**
 * ConnectionTab — Tab 1 of the stream builder.
 *
 * Account selection, base URL, HTTP method, and read-only auth type display.
 */

import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Select from '@splunk/react-ui/Select';
import Text from '@splunk/react-ui/Text';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';

import { useBuilder } from '../../context/BuilderContext';
import { listAccounts } from '../../services/splunk-api';
import type { AccountSummary } from '../../types';

const TabContent = styled.div`
    padding: ${variables.spacingLarge} 0;
    max-width: 640px;
`;

const AuthHint = styled.div`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-top: ${variables.spacingXSmall};
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
        <TabContent>
            <ControlGroup label="Account" help="Select the authentication account for this API.">
                {loadingAccounts ? (
                    <WaitSpinner />
                ) : (
                    <Select
                        value={state.account}
                        onChange={(e: any, { value }: any) => setField('account', value)}
                    >
                        <Select.Option label="(none)" value="" />
                        {accounts.map((acct) => (
                            <Select.Option
                                key={acct.name}
                                label={acct.name}
                                value={acct.name}
                            />
                        ))}
                    </Select>
                )}
                {selectedAccount && (
                    <AuthHint>
                        Auth type: {selectedAccount.authType}
                    </AuthHint>
                )}
            </ControlGroup>

            <ControlGroup label="Base URL" help="The root URL of the API (e.g. https://api.example.com/v1).">
                <Text
                    value={state.baseUrl}
                    onChange={(e: any, { value }: any) => setField('baseUrl', value)}
                    placeholder="https://api.example.com/v1"
                />
            </ControlGroup>

            <ControlGroup label="HTTP Method" help="HTTP method to use for API requests.">
                <Select
                    value={state.httpMethod}
                    onChange={(e: any, { value }: any) => setField('httpMethod', value)}
                >
                    <Select.Option label="GET" value="GET" />
                    <Select.Option label="POST" value="POST" />
                </Select>
            </ControlGroup>
        </TabContent>
    );
}

export default ConnectionTab;
