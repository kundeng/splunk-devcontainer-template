/**
 * TestPanel — Test connection button with loading state and result display.
 */

import React, { useCallback } from 'react';
import Button from '@splunk/react-ui/Button';
import Message from '@splunk/react-ui/Message';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';

import { useBuilder } from '../../context/BuilderContext';
import { testConnection } from '../../services/splunk-api';
import { formStateToManifest } from '../../services/manifest-serializer';

export const TestPanel: React.FC = () => {
    const { state, dispatch } = useBuilder();

    const handleTest = useCallback(async () => {
        // Warn for non-GET methods
        if (state.httpMethod !== 'GET') {
            const confirmed = window.confirm(
                `This test will make a ${state.httpMethod} request to the API. Proceed?`
            );
            if (!confirmed) return;
        }

        dispatch({ type: 'SET_TEST_LOADING', loading: true });

        try {
            const manifest = formStateToManifest(state);
            const result = await testConnection(manifest, state.account, state.baseUrl);
            dispatch({ type: 'SET_TEST_RESULT', result });
        } catch (e: any) {
            dispatch({
                type: 'SET_TEST_RESULT',
                result: {
                    status: 'error',
                    message: e.message || 'Test connection failed',
                    records: [],
                    recordCount: 0,
                    validationResults: state.validationResults,
                    detectedFields: [],
                    timestampCandidates: [],
                },
            });
        }
    }, [state, dispatch]);

    const isDisabled = !state.account || !state.baseUrl;

    return (
        <div>
            <Button
                label={state.testLoading ? 'Testing...' : 'Test Connection'}
                appearance="primary"
                onClick={handleTest}
                disabled={state.testLoading || isDisabled}
            />
            {state.testLoading && <WaitSpinner style={{ marginLeft: 12 }} />}

            {isDisabled && (
                <Message type="info" style={{ marginTop: 12 }}>
                    Select an account and enter a base URL before testing.
                </Message>
            )}

            {state.testResult && !state.testLoading && (
                <div style={{ marginTop: 12 }}>
                    {state.testResult.status === 'success' ? (
                        <Message type="success">
                            Connection successful — {state.testResult.recordCount} record(s) fetched.
                        </Message>
                    ) : (
                        <Message type="error">
                            {state.testResult.message || 'Connection failed.'}
                            {state.testResult.message?.includes('401') && (
                                <span>
                                    {' '}Check credentials in <strong>Configuration &rarr; Accounts</strong>.
                                </span>
                            )}
                        </Message>
                    )}
                </div>
            )}

            {state.testResult?.validationResults &&
                state.testResult.validationResults.length > 0 && (
                    <div style={{ marginTop: 12 }}>
                        {state.testResult.validationResults.map((v, i) => (
                            <Message
                                key={i}
                                type={v.severity === 'error' ? 'error' : 'warning'}
                            >
                                <strong>{v.type}</strong> at {v.path}: {v.message}
                            </Message>
                        ))}
                    </div>
                )}
        </div>
    );
};
