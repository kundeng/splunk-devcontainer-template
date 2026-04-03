import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TagInput } from './TagInput';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function renderTagInput(overrides: Partial<React.ComponentProps<typeof TagInput>> = {}) {
    const defaultProps = {
        tags: [],
        allTags: [],
        onChange: jest.fn(),
        ...overrides,
    };
    const result = render(<TagInput {...defaultProps} />);
    return { ...result, onChange: defaultProps.onChange };
}

function getInput() {
    return screen.getByRole('textbox');
}

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */

describe('TagInput', () => {
    it('renders existing tags as chips', () => {
        renderTagInput({ tags: ['devops', 'security'] });
        expect(screen.getByText('devops')).toBeInTheDocument();
        expect(screen.getByText('security')).toBeInTheDocument();
    });

    it('typing + Enter adds a new tag', async () => {
        const user = userEvent.setup();
        const { onChange } = renderTagInput({ tags: [] });

        await user.click(getInput());
        await user.type(getInput(), 'newTag{Enter}');

        expect(onChange).toHaveBeenCalledWith(['newTag']);
    });

    it('typing comma adds a new tag', async () => {
        const user = userEvent.setup();
        const { onChange } = renderTagInput({ tags: [] });

        await user.click(getInput());
        await user.type(getInput(), 'newTag,');

        expect(onChange).toHaveBeenCalledWith(['newTag']);
    });

    it('removes a chip when remove button is clicked', async () => {
        const user = userEvent.setup();
        const { onChange } = renderTagInput({ tags: ['devops', 'security'] });

        // Splunk Chip renders a remove button; find all remove buttons
        const removeButtons = screen.getAllByRole('button');
        await user.click(removeButtons[0]);

        expect(onChange).toHaveBeenCalledWith(['security']);
    });

    it('does not add duplicate tags', async () => {
        const user = userEvent.setup();
        const { onChange } = renderTagInput({ tags: ['devops'] });

        await user.click(getInput());
        await user.type(getInput(), 'devops{Enter}');

        // onChange should not be called since 'devops' already exists
        expect(onChange).not.toHaveBeenCalled();
    });

    it('backspace on empty input removes last tag', async () => {
        const user = userEvent.setup();
        const { onChange } = renderTagInput({ tags: ['devops', 'security'] });

        await user.click(getInput());
        await user.keyboard('{Backspace}');

        expect(onChange).toHaveBeenCalledWith(['devops']);
    });

    it('shows autocomplete suggestions when typing', async () => {
        const user = userEvent.setup();
        renderTagInput({
            tags: [],
            allTags: ['devops', 'dev-tools', 'security'],
        });

        await user.click(getInput());
        await user.type(getInput(), 'dev');

        expect(screen.getByText('devops')).toBeInTheDocument();
        expect(screen.getByText('dev-tools')).toBeInTheDocument();
        expect(screen.queryByText('security')).not.toBeInTheDocument();
    });

    it('shows placeholder when no tags exist', () => {
        renderTagInput({ tags: [] });
        expect(screen.getByPlaceholderText('Add tags...')).toBeInTheDocument();
    });

    it('hides placeholder when tags exist', () => {
        renderTagInput({ tags: ['devops'] });
        expect(screen.queryByPlaceholderText('Add tags...')).not.toBeInTheDocument();
    });
});
