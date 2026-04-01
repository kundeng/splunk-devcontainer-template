/**
 * TestPreviewTab — Tab 5: Test connection, preview records, detect fields.
 */

import React from 'react';
import ColumnLayout from '@splunk/react-ui/ColumnLayout';
import Heading from '@splunk/react-ui/Heading';

import { useBuilder } from '../../context/BuilderContext';
import { TestPanel } from '../test/TestPanel';
import { RecordPreview } from '../test/RecordPreview';
import { FieldDetector } from '../test/FieldDetector';

const TestPreviewTab: React.FC = () => {
    const { state } = useBuilder();
    const hasResults = state.testResult && state.testResult.status === 'success';

    return (
        <div>
            <TestPanel />

            {hasResults && (
                <>
                    <Heading level={3} style={{ marginTop: '24px' }}>Results</Heading>
                    <ColumnLayout gutter={16}>
                        <ColumnLayout.Row>
                            <ColumnLayout.Column span={8}>
                                <RecordPreview records={state.testResult!.records} />
                            </ColumnLayout.Column>
                            <ColumnLayout.Column span={4}>
                                <FieldDetector fields={state.testResult!.detectedFields} />
                            </ColumnLayout.Column>
                        </ColumnLayout.Row>
                    </ColumnLayout>
                </>
            )}
        </div>
    );
};

export default TestPreviewTab;
