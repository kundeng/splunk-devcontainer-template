/**
 * ScheduleTagsTab — Tab 4: Interval, tags, debug toggle, enable/disable.
 */

import React from 'react';
import ControlGroup from '@splunk/react-ui/ControlGroup';
import Text from '@splunk/react-ui/Text';
import Switch from '@splunk/react-ui/Switch';
import Number from '@splunk/react-ui/Number';
import Heading from '@splunk/react-ui/Heading';

import { useBuilder } from '../../context/BuilderContext';

const ScheduleTagsTab: React.FC = () => {
    const { state, dispatch } = useBuilder();

    const setField = (path: string) => (e: any, data: any) => {
        dispatch({ type: 'SET_FIELD', path, value: data.value });
    };

    const setSwitch = (path: string) => (e: any, data: any) => {
        dispatch({ type: 'SET_FIELD', path, value: data.selected });
    };

    return (
        <div>
            <Heading level={3}>Schedule</Heading>

            <ControlGroup label="Interval (seconds)" help="Collection interval in seconds (10–86400).">
                <Number
                    value={state.interval}
                    onChange={setField('interval')}
                    min={10}
                    max={86400}
                    step={10}
                />
            </ControlGroup>

            <Heading level={3} style={{ marginTop: '24px' }}>Organization</Heading>

            <ControlGroup label="Tags" help="Comma-separated tags for organizing streams (e.g. devops, security).">
                <Text
                    value={state.tags}
                    onChange={setField('tags')}
                    placeholder="devops, security, monitoring"
                />
            </ControlGroup>

            <Heading level={3} style={{ marginTop: '24px' }}>Controls</Heading>

            <ControlGroup label="Enabled" help="Enable or disable this stream.">
                <Switch
                    selected={state.enabled}
                    onChange={setSwitch('enabled')}
                    appearance="toggle"
                />
            </ControlGroup>

            <ControlGroup label="Debug Logging" help="Enable verbose debug logging for this stream. Adds method entry/exit, URL, and stream slice details to logs.">
                <Switch
                    selected={state.debug}
                    onChange={setSwitch('debug')}
                    appearance="toggle"
                />
            </ControlGroup>
        </div>
    );
};

export default ScheduleTagsTab;
