/**
 * ValidationBanner — Renders filtered validation warnings/errors.
 *
 * Filters ValidationResult[] by pathPrefix and renders each as a
 * Splunk Message component with the appropriate severity.
 */

import React, { useMemo } from 'react';
import styled from 'styled-components';
import { variables } from '@splunk/themes';
import Message from '@splunk/react-ui/Message';

import type { ValidationResult } from '../../types';

const BannerContainer = styled.div`
    margin-bottom: ${variables.spacingSmall};
`;

export interface ValidationBannerProps {
    results: ValidationResult[];
    pathPrefix: string;
}

export function ValidationBanner({ results, pathPrefix }: ValidationBannerProps) {
    const filtered = useMemo(
        () => results.filter((r) => r.path.startsWith(pathPrefix)),
        [results, pathPrefix]
    );

    if (filtered.length === 0) return null;

    return (
        <BannerContainer>
            {filtered.map((result, idx) => (
                <Message
                    key={`${result.path}-${idx}`}
                    type={result.severity === 'info' ? 'info' : result.severity}
                >
                    {result.message}
                </Message>
            ))}
        </BannerContainer>
    );
}

export default ValidationBanner;
