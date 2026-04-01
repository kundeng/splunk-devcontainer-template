/**
 * SplunkOutputTab — Tab 3: Index, sourcetype, source, host, and timestamp picker.
 */

import React, { useEffect, useState } from 'react';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Select from '@splunk/react-ui/Select';
import Text from '@splunk/react-ui/Text';
import Message from '@splunk/react-ui/Message';
import Heading from '@splunk/react-ui/Heading';

import { useBuilder } from '../../context/BuilderContext';
import { listIndexes } from '../../services/splunk-api';
import { TimestampPicker } from '../test/TimestampPicker';

const SplunkOutputTab: React.FC = () => {
    const { state, dispatch } = useBuilder();
    const [indexes, setIndexes] = useState<string[]>([]);
    const [indexError, setIndexError] = useState<string | null>(null);

    useEffect(() => {
        listIndexes()
            .then(setIndexes)
            .catch((e) => setIndexError(e.message));
    }, []);

    const setField = (path: string) => (e: any, data: any) => {
        dispatch({ type: 'SET_FIELD', path, value: data.value });
    };

    return (
        <div>
            <Heading level={3}>Splunk Destination</Heading>

            <ControlGroup label="Index" help="Splunk index for collected events.">
                {indexError ? (
                    <Message type="warning">Unable to load indexes: {indexError}</Message>
                ) : (
                    <Select value={state.index} onChange={setField('index')}>
                        {indexes.map((idx) => (
                            <Select.Option key={idx} label={idx} value={idx} />
                        ))}
                    </Select>
                )}
            </ControlGroup>

            <ControlGroup label="Sourcetype" help="Event sourcetype (default: urc:api:json).">
                <Text
                    value={state.sourcetype}
                    onChange={setField('sourcetype')}
                />
            </ControlGroup>

            <ControlGroup label="Source Override" help="Optional. Override the event source field.">
                <Text
                    value={state.sourceOverride}
                    onChange={setField('sourceOverride')}
                    placeholder="Leave empty for default (urc:<input>:<stream>)"
                />
            </ControlGroup>

            <ControlGroup label="Host Override" help="Optional. Override the event host field.">
                <Text
                    value={state.hostOverride}
                    onChange={setField('hostOverride')}
                    placeholder="Leave empty for default"
                />
            </ControlGroup>

            <Heading level={3} style={{ marginTop: '24px' }}>Event Timestamp</Heading>

            {state.testResult && state.testResult.timestampCandidates.length > 0 ? (
                <TimestampPicker
                    candidates={state.testResult.timestampCandidates}
                    value={state.timestampField}
                    onChange={(field) =>
                        dispatch({ type: 'SET_FIELD', path: 'timestampField', value: field })
                    }
                />
            ) : (
                <Message type="info">
                    Run a test fetch in the <strong>Test &amp; Preview</strong> tab to detect
                    available timestamp fields from your API response.
                </Message>
            )}
        </div>
    );
};

export default SplunkOutputTab;
