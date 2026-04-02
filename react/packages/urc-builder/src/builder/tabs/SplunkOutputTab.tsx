/**
 * SplunkOutputTab — Step 3: Destination + Schedule in a two-column layout.
 *
 * Merges the former SplunkOutput and ScheduleTags tabs into one step.
 */

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import ColumnLayout from '@splunk/react-ui/ColumnLayout';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Select from '@splunk/react-ui/Select';
import Text from '@splunk/react-ui/Text';
import Number from '@splunk/react-ui/Number';
import Message from '@splunk/react-ui/Message';
import CollapsiblePanel from '@splunk/react-ui/CollapsiblePanel';
import Switch from '@splunk/react-ui/Switch';

import { useBuilder } from '../../context/BuilderContext';
import { listIndexes } from '../../services/splunk-api';
import { SECTIONS, FIELDS } from '../../content';
import { SectionHeader } from '../SectionHeader';

const SectionCard = styled.div`
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingMedium} ${variables.spacingLarge};
    margin-bottom: ${variables.spacingMedium};
`;

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

    const setSwitch = (path: string) => (e: any, data: any) => {
        dispatch({ type: 'SET_FIELD', path, value: data.selected });
    };

    return (
        <ColumnLayout gutter={16}>
            <ColumnLayout.Row>
                <ColumnLayout.Column span={7}>
                    <SectionCard>
                        <SectionHeader
                            icon={SECTIONS.destination.icon}
                            title={SECTIONS.destination.title}
                            description={SECTIONS.destination.description}
                        />

                        <ControlGroup label={FIELDS.index.label} help={FIELDS.index.help}>
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

                        <ControlGroup label={FIELDS.sourcetype.label} help={FIELDS.sourcetype.help}>
                            <Text value={state.sourcetype} onChange={setField('sourcetype')} />
                        </ControlGroup>

                        <ControlGroup label={FIELDS.sourceOverride.label} help={FIELDS.sourceOverride.help}>
                            <Text
                                value={state.sourceOverride}
                                onChange={setField('sourceOverride')}
                                placeholder="Leave empty for default"
                            />
                        </ControlGroup>
                    </SectionCard>
                </ColumnLayout.Column>

                <ColumnLayout.Column span={5}>
                    <SectionCard>
                        <SectionHeader
                            icon={SECTIONS.schedule.icon}
                            title={SECTIONS.schedule.title}
                            description={SECTIONS.schedule.description}
                        />

                        <ControlGroup label={FIELDS.interval.label} help={FIELDS.interval.help}>
                            <Number
                                value={state.interval}
                                onChange={setField('interval')}
                                min={10}
                                max={86400}
                                step={10}
                            />
                        </ControlGroup>

                        <ControlGroup label={FIELDS.tags.label} help={FIELDS.tags.help}>
                            <Text
                                value={state.tags}
                                onChange={setField('tags')}
                                placeholder="e.g. devops, security"
                            />
                        </ControlGroup>
                    </SectionCard>

                    <CollapsiblePanel title="Advanced">
                        <ControlGroup label={FIELDS.debug.label} help={FIELDS.debug.help}>
                            <Switch
                                selected={state.debug}
                                onChange={setSwitch('debug')}
                                appearance="toggle"
                            />
                        </ControlGroup>
                        <ControlGroup label={FIELDS.hostOverride.label} help={FIELDS.hostOverride.help}>
                            <Text
                                value={state.hostOverride}
                                onChange={setField('hostOverride')}
                                placeholder="Leave empty for default"
                            />
                        </ControlGroup>
                    </CollapsiblePanel>
                </ColumnLayout.Column>
            </ColumnLayout.Row>
        </ColumnLayout>
    );
};

export default SplunkOutputTab;
