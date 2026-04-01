/**
 * StreamTable — data table displaying all filtered streams with row selection.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import Table from '@splunk/react-ui/Table';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';
import { variables } from '@splunk/themes';
import { useDashboard } from '../context/DashboardContext';

const StatusBadge = styled.span<{ $active: boolean }>`
    color: ${(props) => (props.$active ? variables.accentColorPositive : variables.contentColorMuted)};
    font-weight: 600;
`;

const SpinnerWrapper = styled.div`
    display: flex;
    justify-content: center;
    padding: ${variables.spacingXLarge} 0;
`;

const EmptyMessage = styled.div`
    text-align: center;
    padding: ${variables.spacingXLarge} 0;
    color: ${variables.contentColorMuted};
`;

export default function StreamTable() {
    const { filteredInputs, state, setSelectedRows } = useDashboard();
    const { loading, selectedRows } = state;

    const handleRowClick = useCallback((name: string) => {
        window.location.href = `builder?input=${encodeURIComponent(name)}`;
    }, []);

    const handleCheckboxChange = useCallback(
        (name: string) => {
            if (selectedRows.includes(name)) {
                setSelectedRows(selectedRows.filter((r) => r !== name));
            } else {
                setSelectedRows([...selectedRows, name]);
            }
        },
        [selectedRows, setSelectedRows]
    );

    const handleSelectAll = useCallback(() => {
        if (selectedRows.length === filteredInputs.length) {
            setSelectedRows([]);
        } else {
            setSelectedRows(filteredInputs.map((i) => i.name));
        }
    }, [selectedRows, filteredInputs, setSelectedRows]);

    if (loading) {
        return (
            <SpinnerWrapper>
                <WaitSpinner size="large" />
            </SpinnerWrapper>
        );
    }

    if (filteredInputs.length === 0) {
        return <EmptyMessage>No streams found. Create one to get started.</EmptyMessage>;
    }

    const allSelected = selectedRows.length === filteredInputs.length && filteredInputs.length > 0;

    return (
        <Table stripeRows>
            <Table.Head>
                <Table.HeadCell>
                    <input
                        type="checkbox"
                        checked={allSelected}
                        onChange={handleSelectAll}
                        aria-label="Select all rows"
                    />
                </Table.HeadCell>
                <Table.HeadCell>Name</Table.HeadCell>
                <Table.HeadCell>Tags</Table.HeadCell>
                <Table.HeadCell>Account</Table.HeadCell>
                <Table.HeadCell>Status</Table.HeadCell>
                <Table.HeadCell>Interval</Table.HeadCell>
            </Table.Head>
            <Table.Body>
                {filteredInputs.map((input) => (
                    <Table.Row key={input.name}>
                        <Table.Cell>
                            <input
                                type="checkbox"
                                checked={selectedRows.includes(input.name)}
                                onChange={() => handleCheckboxChange(input.name)}
                                onClick={(e) => e.stopPropagation()}
                                aria-label={`Select ${input.name}`}
                            />
                        </Table.Cell>
                        <Table.Cell
                            onClick={() => handleRowClick(input.name)}
                            style={{ cursor: 'pointer' }}
                        >
                            {input.name}
                        </Table.Cell>
                        <Table.Cell>
                            {input.tags
                                ? input.tags
                                      .split(',')
                                      .map((t) => t.trim())
                                      .filter(Boolean)
                                      .join(', ')
                                : '\u2014'}
                        </Table.Cell>
                        <Table.Cell>{input.account || '\u2014'}</Table.Cell>
                        <Table.Cell>
                            <StatusBadge $active={!input.disabled}>
                                {input.disabled ? 'Disabled' : 'Active'}
                            </StatusBadge>
                        </Table.Cell>
                        <Table.Cell>{input.interval}s</Table.Cell>
                    </Table.Row>
                ))}
            </Table.Body>
        </Table>
    );
}
