/**
 * TagFilter — renders clickable toggle buttons for each unique tag across all inputs.
 */

import React, { useCallback } from 'react';
import styled from 'styled-components';
import Button from '@splunk/react-ui/Button';
import { variables } from '@splunk/themes';
import { useDashboard } from '../context/DashboardContext';

const FilterBar = styled.div`
    display: flex;
    flex-wrap: wrap;
    gap: ${variables.spacingSmall};
    align-items: center;
    margin: ${variables.spacingMedium} 0;
`;

const FilterLabel = styled.span`
    font-size: ${variables.fontSizeSmall};
    color: ${variables.contentColorMuted};
    margin-right: ${variables.spacingXSmall};
`;

export default function TagFilter() {
    const { allTags, state, setSelectedTags } = useDashboard();
    const { selectedTags } = state;

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

    const handleAllClick = useCallback(() => {
        setSelectedTags([]);
    }, [setSelectedTags]);

    if (allTags.length === 0) {
        return null;
    }

    return (
        <FilterBar>
            <FilterLabel>Filter by tag:</FilterLabel>
            <Button
                appearance="toggle"
                selected={selectedTags.length === 0}
                onClick={handleAllClick}
            >
                All
            </Button>
            {allTags.map((tag) => (
                <Button
                    key={tag}
                    appearance="toggle"
                    selected={selectedTags.includes(tag)}
                    onClick={() => handleTagClick(tag)}
                >
                    {tag}
                </Button>
            ))}
        </FilterBar>
    );
}
