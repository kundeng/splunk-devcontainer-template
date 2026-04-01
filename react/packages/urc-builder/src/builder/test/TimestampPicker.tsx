/**
 * TimestampPicker — shows detected datetime fields from test results
 * and lets the user pick which one becomes Splunk _time.
 */

import React from 'react';
import RadioBar from '@splunk/react-ui/RadioBar';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Message from '@splunk/react-ui/Message';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import type { TimestampCandidate } from '../../types';

const CandidateRow = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingMedium};
    padding: ${variables.spacingSmall} 0;
    font-size: ${variables.fontSizeSmall};
`;

const SampleValue = styled.span`
    color: ${variables.contentColorMuted};
    font-family: monospace;
`;

interface TimestampPickerProps {
    candidates: TimestampCandidate[];
    value: string | null;
    onChange: (field: string) => void;
}

export const TimestampPicker: React.FC<TimestampPickerProps> = ({
    candidates,
    value,
    onChange,
}) => {
    if (candidates.length === 0) {
        return (
            <Message type="info">
                No timestamp candidates detected in the sample records.
            </Message>
        );
    }

    return (
        <ControlGroup
            label="Timestamp Field"
            help="Select which field should be used as the Splunk event timestamp (_time)."
        >
            <RadioBar
                value={value || ''}
                onChange={(e, { value: v }) => onChange(v as string)}
            >
                {candidates.map((c) => (
                    <RadioBar.Option key={c.field} value={c.field} label={c.field} />
                ))}
            </RadioBar>
            {value && (
                <div style={{ marginTop: '8px' }}>
                    {candidates
                        .filter((c) => c.field === value)
                        .map((c) => (
                            <CandidateRow key={c.field}>
                                <span>Sample:</span>
                                <SampleValue>{c.sampleValue}</SampleValue>
                                <span>Parsed:</span>
                                <SampleValue>{c.parsedPreview}</SampleValue>
                            </CandidateRow>
                        ))}
                </div>
            )}
        </ControlGroup>
    );
};
