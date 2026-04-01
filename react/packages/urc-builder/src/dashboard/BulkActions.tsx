/**
 * BulkActions — enable, disable, or delete multiple selected streams.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import Button from '@splunk/react-ui/Button';
import { variables } from '@splunk/themes';
import { useDashboard } from '../context/DashboardContext';
import { toggleInput, deleteInput, listInputs } from '../services/splunk-api';

const ActionsBar = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
    padding: ${variables.spacingSmall} ${variables.spacingMedium};
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    margin: ${variables.spacingSmall} 0;
`;

const SelectionCount = styled.span`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-right: ${variables.spacingSmall};
`;

export default function BulkActions() {
    const { state, setInputs, setSelectedRows, setLoading, setError } = useDashboard();
    const { selectedRows } = state;

    const refreshInputs = useCallback(async () => {
        try {
            setLoading(true);
            const inputs = await listInputs();
            setInputs(inputs);
            setSelectedRows([]);
        } catch (err: any) {
            setError(err.message || 'Failed to refresh inputs');
        }
    }, [setInputs, setSelectedRows, setLoading, setError]);

    const handleEnable = useCallback(async () => {
        try {
            setLoading(true);
            await Promise.all(selectedRows.map((name) => toggleInput(name, false)));
            await refreshInputs();
        } catch (err: any) {
            setError(err.message || 'Failed to enable inputs');
        }
    }, [selectedRows, refreshInputs, setLoading, setError]);

    const handleDisable = useCallback(async () => {
        try {
            setLoading(true);
            await Promise.all(selectedRows.map((name) => toggleInput(name, true)));
            await refreshInputs();
        } catch (err: any) {
            setError(err.message || 'Failed to disable inputs');
        }
    }, [selectedRows, refreshInputs, setLoading, setError]);

    const handleDelete = useCallback(async () => {
        const count = selectedRows.length;
        const confirmed = window.confirm(
            `Are you sure you want to delete ${count} stream${count > 1 ? 's' : ''}? This cannot be undone.`
        );
        if (!confirmed) return;

        try {
            setLoading(true);
            await Promise.all(selectedRows.map((name) => deleteInput(name)));
            await refreshInputs();
        } catch (err: any) {
            setError(err.message || 'Failed to delete inputs');
        }
    }, [selectedRows, refreshInputs, setLoading, setError]);

    if (selectedRows.length === 0) {
        return null;
    }

    return (
        <ActionsBar>
            <SelectionCount>
                {selectedRows.length} stream{selectedRows.length > 1 ? 's' : ''} selected
            </SelectionCount>
            <Button appearance="secondary" onClick={handleEnable}>
                Enable Selected
            </Button>
            <Button appearance="secondary" onClick={handleDisable}>
                Disable Selected
            </Button>
            <Button appearance="destructive" onClick={handleDelete}>
                Delete Selected
            </Button>
        </ActionsBar>
    );
}
