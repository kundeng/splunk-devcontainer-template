/**
 * StreamTable — data table with badges, chips, row actions, and empty state hero.
 */

import React, { useCallback, useState } from 'react';
import styled from 'styled-components';
import Table from '@splunk/react-ui/Table';
import Button from '@splunk/react-ui/Button';
import Badge from '@splunk/react-ui/Badge';
import Chip from '@splunk/react-ui/Chip';
import Heading from '@splunk/react-ui/Heading';
import Paragraph from '@splunk/react-ui/Paragraph';
import Menu from '@splunk/react-ui/Menu';
import Dropdown from '@splunk/react-ui/Dropdown';
import Tooltip from '@splunk/react-ui/Tooltip';
import WaitSpinner from '@splunk/react-ui/WaitSpinner';
import { variables } from '@splunk/themes';
import Plus from '@splunk/react-icons/Plus';
import { DASHBOARD } from '../content';
import Pencil from '@splunk/react-icons/Pencil';
import TrashCanCross from '@splunk/react-icons/TrashCanCross';
import DotsThreeVertical from '@splunk/react-icons/DotsThreeVertical';
import ControlPlay from '@splunk/react-icons/ControlPlay';
import ControlPause from '@splunk/react-icons/ControlPause';
import { useDashboard } from '../context/DashboardContext';
import { toggleInput, deleteInput, listInputs } from '../services/splunk-api';

const SpinnerWrapper = styled.div`
    display: flex;
    justify-content: center;
    padding: ${variables.spacingXLarge} 0;
`;

const EmptyHero = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 64px ${variables.spacingXLarge};
    text-align: center;
`;

const HeroIcon = styled.div`
    color: ${variables.contentColorMuted};
    font-size: 48px;
    margin-bottom: ${variables.spacingMedium};
    opacity: 0.4;
`;

const NameLink = styled.span`
    color: ${variables.infoColor};
    font-weight: 600;
    cursor: pointer;
    &:hover {
        text-decoration: underline;
    }
`;

const TagGroup = styled.div`
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
`;

const ActionsCell = styled.div`
    display: flex;
    justify-content: flex-end;
`;

export default function StreamTable() {
    const { filteredInputs, state, setInputs, setSelectedRows, setError } = useDashboard();
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

    const handleToggle = useCallback(async (name: string, disabled: boolean) => {
        try {
            await toggleInput(name, !disabled);
            const inputs = await listInputs();
            setInputs(inputs);
        } catch (err: any) {
            setError(err.message);
        }
    }, [setInputs, setError]);

    const handleDelete = useCallback(async (name: string) => {
        if (!window.confirm(`Delete stream "${name}"? This cannot be undone.`)) return;
        try {
            await deleteInput(name);
            const inputs = await listInputs();
            setInputs(inputs);
            setSelectedRows([]);
        } catch (err: any) {
            setError(err.message);
        }
    }, [setInputs, setSelectedRows, setError]);

    if (loading) {
        return (
            <SpinnerWrapper>
                <WaitSpinner size="large" />
            </SpinnerWrapper>
        );
    }

    if (filteredInputs.length === 0) {
        return (
            <EmptyHero>
                <HeroIcon>{DASHBOARD.cards.total.icon}</HeroIcon>
                <Heading level={2}>{DASHBOARD.emptyState.heading}</Heading>
                <Paragraph style={{ color: variables.contentColorMuted, maxWidth: 420, marginBottom: variables.spacingMedium }}>
                    {DASHBOARD.emptyState.description}
                </Paragraph>
                <Button
                    appearance="primary"
                    icon={<Plus />}
                    label={DASHBOARD.emptyState.cta}
                    onClick={() => { window.location.href = 'builder'; }}
                />
            </EmptyHero>
        );
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
                <Table.HeadCell>Status</Table.HeadCell>
                <Table.HeadCell>Tags</Table.HeadCell>
                <Table.HeadCell>Account</Table.HeadCell>
                <Table.HeadCell>Interval</Table.HeadCell>
                <Table.HeadCell align="right">Actions</Table.HeadCell>
            </Table.Head>
            <Table.Body>
                {filteredInputs.map((input) => {
                    const tags = input.tags
                        ? input.tags.split(',').map((t) => t.trim()).filter(Boolean)
                        : [];

                    return (
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
                            <Table.Cell>
                                <NameLink onClick={() => handleRowClick(input.name)}>
                                    {input.name}
                                </NameLink>
                            </Table.Cell>
                            <Table.Cell>
                                <Tooltip content={input.disabled ? 'Stream is paused' : `Collecting every ${input.interval}s`}>
                                    <Badge
                                        label={input.disabled ? 'Disabled' : 'Active'}
                                        backgroundColor={input.disabled ? variables.contentColorMuted : variables.accentColorPositive}
                                        foregroundColor="#fff"
                                    />
                                </Tooltip>
                            </Table.Cell>
                            <Table.Cell>
                                {tags.length > 0 ? (
                                    <TagGroup>
                                        {tags.map((tag) => (
                                            <Chip key={tag} appearance="outline">{tag}</Chip>
                                        ))}
                                    </TagGroup>
                                ) : (
                                    <span style={{ color: variables.contentColorMuted }}>{'\u2014'}</span>
                                )}
                            </Table.Cell>
                            <Table.Cell>{input.account || '\u2014'}</Table.Cell>
                            <Table.Cell>{input.interval}s</Table.Cell>
                            <Table.Cell>
                                <ActionsCell onClick={(e) => e.stopPropagation()}>
                                    <Dropdown
                                        toggle={
                                            <Button
                                                appearance="subtle"
                                                icon={<DotsThreeVertical />}
                                                aria-label={`Actions for ${input.name}`}
                                            />
                                        }
                                    >
                                        <Menu>
                                            <Menu.Item
                                                icon={<Pencil />}
                                                onClick={() => handleRowClick(input.name)}
                                            >
                                                Edit
                                            </Menu.Item>
                                            <Menu.Item
                                                icon={input.disabled ? <ControlPlay /> : <ControlPause />}
                                                onClick={() => handleToggle(input.name, input.disabled)}
                                            >
                                                {input.disabled ? 'Enable' : 'Disable'}
                                            </Menu.Item>
                                            <Menu.Divider />
                                            <Menu.Item
                                                icon={<TrashCanCross />}
                                                onClick={() => handleDelete(input.name)}
                                            >
                                                Delete
                                            </Menu.Item>
                                        </Menu>
                                    </Dropdown>
                                </ActionsCell>
                            </Table.Cell>
                        </Table.Row>
                    );
                })}
            </Table.Body>
        </Table>
    );
}
