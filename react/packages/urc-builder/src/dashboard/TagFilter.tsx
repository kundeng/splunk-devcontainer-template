/**
 * TagFilter — status filter (RadioBar) + tag toggle chips.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import RadioBar from '@splunk/react-ui/RadioBar';
import Chip from '@splunk/react-ui/Chip';
import { variables } from '@splunk/themes';
import { useDashboard } from '../context/DashboardContext';

const FilterRow = styled.div`
    display: flex;
    flex-wrap: wrap;
    gap: ${variables.spacingMedium};
    align-items: center;
    margin: ${variables.spacingSmall} 0 ${variables.spacingMedium} 0;
`;

const FilterGroup = styled.div`
    display: flex;
    flex-wrap: wrap;
    gap: ${variables.spacingXSmall};
    align-items: center;
`;

const FilterLabel = styled.span`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-right: 4px;
`;

const Separator = styled.div`
    width: 1px;
    height: 24px;
    background: ${variables.borderColor};
    margin: 0 ${variables.spacingXSmall};
`;

export default function TagFilter() {
    const { allTags, state, setSelectedTags, setSelectedStatus } = useDashboard();
    const { selectedTags, selectedStatus } = state;

    const handleTagClick = useCallback(
        (tag: string) => {
            if (selectedTags.includes(tag)) {
                setSelectedTags(selectedTags.filter((t) => t !== tag));
            } else {
                setSelectedTags([...selectedTags, tag]);
            }
        },
        [selectedTags, setSelectedTags]
    );

    const handleStatusChange = useCallback(
        (_e: any, { value }: any) => {
            setSelectedStatus(value);
        },
        [setSelectedStatus]
    );

    return (
        <FilterRow>
            <FilterGroup>
                <FilterLabel>Status:</FilterLabel>
                <RadioBar value={selectedStatus} onChange={handleStatusChange}>
                    <RadioBar.Option value="all" label="All" />
                    <RadioBar.Option value="active" label="Active" />
                    <RadioBar.Option value="disabled" label="Disabled" />
                </RadioBar>
            </FilterGroup>

            {allTags.length > 0 && (
                <>
                    <Separator />
                    <FilterGroup>
                        <FilterLabel>Tags:</FilterLabel>
                        <Chip
                            appearance={selectedTags.length === 0 ? 'info' : 'outline'}
                            onClick={() => setSelectedTags([])}
                        >
                            All
                        </Chip>
                        {allTags.map((tag) => (
                            <Chip
                                key={tag}
                                appearance={selectedTags.includes(tag) ? 'info' : 'outline'}
                                onClick={() => handleTagClick(tag)}
                            >
                                {tag}
                            </Chip>
                        ))}
                    </FilterGroup>
                </>
            )}
        </FilterRow>
    );
}
