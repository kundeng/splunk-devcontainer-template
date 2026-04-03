import React from 'react';
import layout from '@splunk/react-page';
import { getUserTheme } from '@splunk/splunk-utils/themes';
import { StreamBuilder } from '@splunk/urc-builder';

import { StyledContainer } from './Styles';

// Read input name from URL query params for edit mode or clone mode
const params = new URLSearchParams(window.location.search);
const inputName = params.get('input') || undefined;
const cloneName = params.get('clone') || undefined;

getUserTheme()
    .then((theme) => {
        layout(
            <StyledContainer>
                <StreamBuilder inputName={inputName} cloneName={cloneName} />
            </StyledContainer>,
            {
                theme,
            }
        );
    })
    .catch((e) => {
        const errorEl = document.createElement('span');
        errorEl.innerHTML = e;
        document.body.appendChild(errorEl);
    });
