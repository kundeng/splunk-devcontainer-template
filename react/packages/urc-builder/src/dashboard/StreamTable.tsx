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
import ChevronUp from '@splunk/react-icons/ChevronUp';
import ChevronDown from '@splunk/react-icons/ChevronDown';
import { DASHBOARD } from '../content';
import { useSort } from '../hooks/useSort';
import { usePagination } from '../hooks/usePagination';
import Select from '@splunk/react-ui/Select';
import Pencil from '@splunk/react-icons/Pencil';
import TrashCanCross from '@splunk/react-icons/TrashCanCross';
import DotsThreeVertical from '@splunk/react-icons/DotsThreeVertical';
import ControlPlay from '@splunk/react-icons/ControlPlay';
import ControlPause from '@splunk/react-icons/ControlPause';
import FilePlus from '@splunk/react-icons/FilePlus';
import CrossCircle from '@splunk/react-icons/CrossCircle';
import FileText from '@splunk/react-icons/FileText';
import { useDashboard } from '../context/DashboardContext';
import { toggleInput, deleteInput, listInputs } from '../services/splunk-api';
import { exportStreams } from '../utils/export-import';
import { downloadFile } from '../utils/download';

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
    margin-bottom: ${variables.spacingMedium};
    opacity: 0.3;
    & svg {
        width: 56px;
        height: 56px;
    }
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

const SortableHeader = styled.span`
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    user-select: none;
    &:hover {
        color: ${variables.infoColor};
    }
`;

const SortIcon = styled.span`
    display: inline-flex;
    align-items: center;
    font-size: 12px;
`;

const PaginationBar = styled.div`
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: ${variables.spacingSmall} 0;
    margin-top: ${variables.spacingSmall};
`;

const PageInfo = styled.span`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
`;

const PageControls = styled.div`
    display: flex;
    align-items: center;
    gap: ${variables.spacingSmall};
`;

function relativeTime(isoDate: string | null): string {
    if (!isoDate) return 'Never';
    const diff = Date.now() - new Date(isoDate).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

function SortHeader({ label, sortKey, activeSortKey, sortDir, onSort }: {
    label: string;
    sortKey: string;
    activeSortKey: string | null;
    sortDir: 'asc' | 'desc';
    onSort: (key: string) => void;
}) {
    const isActive = activeSortKey === sortKey;
    return (
        <SortableHeader onClick={() => onSort(sortKey)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSort(sortKey); }}>
            {label}
            {isActive && (
                <SortIcon>
                    {sortDir === 'asc' ? <ChevronUp width={14} height={14} /> : <ChevronDown width={14} height={14} />}
                </SortIcon>
            )}
        </SortableHeader>
    );
}

export default function StreamTable({ items }: { items?: import('../types').InputSummary[] } = {}) {
    const { filteredInputs, state, setInputs, setSelectedRows, setError, healthMap } = useDashboard();
    const displayItems = items || filteredInputs;
    const { loading, selectedRows } = state;
    const { sortedItems, sortKey, sortDir, onSort } = useSort(displayItems);
    const { pagedItems, page, pageSize, totalPages, setPage, setPageSize } = usePagination(sortedItems);
    const showPagination = displayItems.length > pageSize;

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

    if (displayItems.length === 0) {
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

    const allSelected = selectedRows.length === displayItems.length && displayItems.length > 0;

    return (
        <>
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
                <Table.HeadCell><SortHeader label="Name" sortKey="name" activeSortKey={sortKey} sortDir={sortDir} onSort={onSort} /></Table.HeadCell>
                <Table.HeadCell><SortHeader label="Status" sortKey="disabled" activeSortKey={sortKey} sortDir={sortDir} onSort={onSort} /></Table.HeadCell>
                <Table.HeadCell>Tags</Table.HeadCell>
                <Table.HeadCell><SortHeader label="Account" sortKey="account" activeSortKey={sortKey} sortDir={sortDir} onSort={onSort} /></Table.HeadCell>
                <Table.HeadCell><SortHeader label="Interval" sortKey="interval" activeSortKey={sortKey} sortDir={sortDir} onSort={onSort} /></Table.HeadCell>
                <Table.HeadCell>Last Run</Table.HeadCell>
                <Table.HeadCell align="right">Actions</Table.HeadCell>
            </Table.Head>
            <Table.Body>
                {pagedItems.map((input) => {
                    const tags = input.tags
                        ? input.tags.split(',').map((t) => t.trim()).filter(Boolean)
                        : [];
                    const health = healthMap[input.name];
                    const hasError = health?.lastStatus === 'error' && !input.disabled;

                    // Determine status label and color
                    let statusLabel = 'Active';
                    let statusColor = variables.accentColorPositive;
                    let statusTooltip = `Collecting every ${input.interval}s`;
                    if (input.disabled) {
                        statusLabel = 'Disabled';
                        statusColor = variables.contentColorMuted;
                        statusTooltip = 'Stream is paused';
                    } else if (hasError) {
                        statusLabel = 'Error';
                        statusColor = '#d41f1c';
                        statusTooltip = health?.lastError || 'Last collection had errors';
                    }

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
                                <Tooltip content={statusTooltip}>
                                    <Badge
                                        label={statusLabel}
                                        backgroundColor={statusColor}
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
                                {health ? (
                                    <Tooltip content={hasError ? `Error: ${health.lastError}` : `${health.lastRecordCount} records`}>
                                        <span style={{ color: hasError ? '#d41f1c' : undefined }}>
                                            {hasError && <CrossCircle width={14} height={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />}
                                            {relativeTime(health.lastRun)}
                                        </span>
                                    </Tooltip>
                                ) : (
                                    <span style={{ color: variables.contentColorMuted }}>{'—'}</span>
                                )}
                            </Table.Cell>
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
                                                icon={<FilePlus />}
                                                onClick={() => { window.location.href = `builder?clone=${encodeURIComponent(input.name)}`; }}
                                            >
                                                Clone
                                            </Menu.Item>
                                            <Menu.Item
                                                icon={<FileText />}
                                                onClick={() => {
                                                    const yamlContent = exportStreams([input]);
                                                    downloadFile(yamlContent, `${input.name}.yaml`);
                                                }}
                                            >
                                                Export
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
        {showPagination && (
            <PaginationBar>
                <PageInfo>
                    Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, displayItems.length)} of {displayItems.length}
                </PageInfo>
                <PageControls>
                    <Select
                        value={pageSize}
                        onChange={(_e: any, { value }: any) => setPageSize(Number(value))}
                        style={{ width: 80 }}
                        aria-label="Page size"
                    >
                        <Select.Option value={10} label="10" />
                        <Select.Option value={25} label="25" />
                        <Select.Option value={50} label="50" />
                    </Select>
                    <Button
                        appearance="secondary"
                        label="Prev"
                        onClick={() => setPage(page - 1)}
                        disabled={page === 0}
                    />
                    <PageInfo>Page {page + 1} of {totalPages}</PageInfo>
                    <Button
                        appearance="secondary"
                        label="Next"
                        onClick={() => setPage(page + 1)}
                        disabled={page >= totalPages - 1}
                    />
                </PageControls>
            </PaginationBar>
        )}
        </>
    );
}
