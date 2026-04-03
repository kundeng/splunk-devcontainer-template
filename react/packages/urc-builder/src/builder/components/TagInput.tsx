/**
 * TagInput -- chip-based tag input with autocomplete suggestions.
 *
 * Tags are rendered as removable Chip components. The text input supports:
 * - Enter or comma to add the current value as a tag
 * - Backspace on empty input to remove the last tag
 * - Autocomplete dropdown filtered from `allTags`
 */

import React, { useState, useCallback, useRef } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Chip from '@splunk/react-ui/Chip';
import Text from '@splunk/react-ui/Text';
import { getAutocompleteSuggestions } from '../../utils/tag-utils';

const Container = styled.div`
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    padding: ${variables.spacingXSmall};
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: center;
    cursor: text;
    min-height: 36px;
`;

const SuggestionsDropdown = styled.div`
    position: absolute;
    z-index: 100;
    background: ${variables.backgroundColorPage};
    border: 1px solid ${variables.borderColor};
    border-radius: ${variables.borderRadius};
    margin-top: 2px;
    max-height: 150px;
    overflow-y: auto;
    min-width: 160px;
`;

const SuggestionItem = styled.div<{ $highlighted?: boolean }>`
    padding: ${variables.spacingXSmall} ${variables.spacingSmall};
    cursor: pointer;
    background: ${(p) => (p.$highlighted ? 'rgba(0, 0, 0, 0.05)' : 'transparent')};
    font-size: ${variables.fontSizeSmall};
    &:hover {
        background: rgba(0, 0, 0, 0.05);
    }
`;

const InputWrapper = styled.div`
    position: relative;
    flex: 1;
    min-width: 100px;
`;

interface TagInputProps {
    tags: string[];
    allTags: string[];
    onChange: (tags: string[]) => void;
}

export function TagInput({ tags, allTags, onChange }: TagInputProps) {
    const [inputValue, setInputValue] = useState('');
    const [showSuggestions, setShowSuggestions] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const suggestions = getAutocompleteSuggestions(inputValue, allTags, tags);

    const addTag = useCallback(
        (tag: string) => {
            const trimmed = tag.trim();
            if (trimmed && !tags.includes(trimmed)) {
                onChange([...tags, trimmed]);
            }
            setInputValue('');
            setShowSuggestions(false);
        },
        [tags, onChange]
    );

    const removeTag = useCallback(
        (tag: string) => {
            onChange(tags.filter((t) => t !== tag));
        },
        [tags, onChange]
    );

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if ((e.key === 'Enter' || e.key === ',') && inputValue.trim()) {
                e.preventDefault();
                addTag(inputValue);
            } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
                removeTag(tags[tags.length - 1]);
            } else if (e.key === 'Escape') {
                setShowSuggestions(false);
            }
        },
        [inputValue, tags, addTag, removeTag]
    );

    const handleInputChange = useCallback(
        (_e: unknown, { value }: { value: string }) => {
            if (value.includes(',')) {
                const parts = value.split(',');
                parts.slice(0, -1).forEach((p: string) => {
                    if (p.trim()) addTag(p);
                });
                setInputValue(parts[parts.length - 1]);
            } else {
                setInputValue(value);
            }
            setShowSuggestions(true);
        },
        [addTag]
    );

    return (
        <div>
            <Container onClick={() => inputRef.current?.focus()}>
                {tags.map((tag) => (
                    <Chip key={tag} appearance="outline" onRequestRemove={() => removeTag(tag)}>
                        {tag}
                    </Chip>
                ))}
                <InputWrapper>
                    <Text
                        inputRef={inputRef}
                        value={inputValue}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setShowSuggestions(true)}
                        onBlur={() => {
                            // Delay to allow clicking suggestions
                            setTimeout(() => setShowSuggestions(false), 200);
                        }}
                        placeholder={tags.length === 0 ? 'Add tags...' : ''}
                        inline
                    />
                    {showSuggestions && suggestions.length > 0 && (
                        <SuggestionsDropdown>
                            {suggestions.map((s) => (
                                <SuggestionItem
                                    key={s}
                                    onMouseDown={(e) => {
                                        e.preventDefault();
                                        addTag(s);
                                    }}
                                >
                                    {s}
                                </SuggestionItem>
                            ))}
                        </SuggestionsDropdown>
                    )}
                </InputWrapper>
            </Container>
        </div>
    );
}

export default TagInput;
