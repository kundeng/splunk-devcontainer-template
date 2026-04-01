/**
 * RecordPreview — renders sample records as formatted JSON.
 */

import React from 'react';
import Code from '@splunk/react-ui/Code';
import Heading from '@splunk/react-ui/Heading';
import Message from '@splunk/react-ui/Message';

interface RecordPreviewProps {
    records: { stream: string; data: Record<string, any> }[];
}

export const RecordPreview: React.FC<RecordPreviewProps> = ({ records }) => {
    if (records.length === 0) {
        return <Message type="info">No records returned.</Message>;
    }

    // Show up to 5 records
    const display = records.slice(0, 5);

    return (
        <div>
            <Heading level={4}>Sample Records ({records.length} total)</Heading>
            {display.map((rec, i) => (
                <div key={i} style={{ marginBottom: 12 }}>
                    <Code language="json" value={JSON.stringify(rec.data, null, 2)} />
                </div>
            ))}
        </div>
    );
};
